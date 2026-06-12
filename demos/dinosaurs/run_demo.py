#!/usr/bin/env python3
"""
Dinosaur fossil occurrence demo for GeneticMarkov.

This is a non-astronomy demo using real PBDB occurrence data. It fits a deliberately
simple toy diversity/extinction model to binned Dinosauria fossil occurrence counts.

It exercises the current package machinery:
- mixed categorical + continuous individuals
- DEAP population / fitness handling
- tournament selection
- fitness-weighted crossover
- adaptive Gaussian mutation
- duplicate prevention
- DEMC hybrid moves during GA generations
- Voronoi sparse-region exploration using geneticmarkov.exploration
- SMC-DEMC posterior refinement using geneticmarkov.smc_demc
- CSV/NPZ/PNG outputs for the GA+DEMC workflow

This is a package demonstration, not a paleontology result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, tools

import matplotlib.pyplot as plt

try:
    import corner
    HAS_CORNER = True
except Exception:
    HAS_CORNER = False

from geneticmarkov.exploration import voronoi_explore_dearths
from geneticmarkov.smc_demc import Bound, run_smc_demc
from geneticmarkov.operators import (
    deduplicate_population,
    fitness_scale,
    reflect_scalar,
    tournament_select,
)


PBDB_DINOSAURIA_CSV = (
    "https://paleobiodb.org/data1.2/occs/list.csv"
    "?base_name=Dinosauria"
    "&show=class,classext,ident,phylo,time"
    "&limit=all"
)


CATEGORICAL_NAMES = [
    "clade_idx",
    "model_idx",
    "likelihood_idx",
    "sampling_idx",
    "extinction_idx",
]

CONTINUOUS_NAMES = [
    "amplitude",
    "baseline",
    "trend",
    "peak_time",
    "peak_width",
    "extinction_time",
    "extinction_width",
    "extinction_depth",
    "sampling_slope",
    "overdispersion",
]

PARAM_COLUMNS = CATEGORICAL_NAMES + CONTINUOUS_NAMES


CLADE_OPTIONS = [
    "Dinosauria",
    "Theropoda",
    "Sauropodomorpha",
    "Ornithischia",
]

MODEL_OPTIONS = [
    "gaussian_pulse",
    "double_pulse",
    "logistic_decline",
]

LIKELIHOOD_OPTIONS = [
    "poisson",
    "negative_binomial",
    "chi2",
]

SAMPLING_OPTIONS = [
    "raw",
    "sqrt_corrected",
    "log_corrected",
]

EXTINCTION_OPTIONS = [
    "none",
    "kt_fixed",
    "free_pulse",
]


@dataclass
class DemoConfig:
    output: Path
    data_dir: Path
    popsize: int = 96
    generations: int = 80
    seed: int = 42
    bin_width: float = 5.0
    min_ma: float = 50.0
    max_ma: float = 250.0
    demc_fraction: float = 0.40
    demc_moves_per_gen: int = 1
    exploration_steps: int = 32
    output_interval: int = 10


class DinosaurProblem:
    """
    Toy fossil occurrence fitting problem.

    Individual layout:
    0-4 categorical:
        clade_idx, model_idx, likelihood_idx, sampling_idx, extinction_idx
    5-14 continuous:
        amplitude, baseline, trend, peak_time, peak_width,
        extinction_time, extinction_width, extinction_depth,
        sampling_slope, overdispersion
    """

    def __init__(self, counts_df: pd.DataFrame, config: DemoConfig):
        self.config = config
        self.counts_df = counts_df.copy()
        self.time_ma = counts_df["time_ma"].to_numpy(float)
        self.raw_counts = counts_df["count"].to_numpy(float)

        self.clade_counts = {}
        for clade in CLADE_OPTIONS:
            col = f"count_{clade}"
            if col in counts_df:
                self.clade_counts[clade] = counts_df[col].to_numpy(float)
            else:
                self.clade_counts[clade] = self.raw_counts.copy()

        self.categorical_indices = [0, 1, 2, 3, 4]
        self.continuous_indices = list(range(5, 15))

        self.index_to_param_map = {
            0: "clade_options",
            1: "model_options",
            2: "likelihood_options",
            3: "sampling_options",
            4: "extinction_options",
            5: "amplitude",
            6: "baseline",
            7: "trend",
            8: "peak_time",
            9: "peak_width",
            10: "extinction_time",
            11: "extinction_width",
            12: "extinction_depth",
            13: "sampling_slope",
            14: "overdispersion",
        }

        self.clade_options = CLADE_OPTIONS
        self.model_options = MODEL_OPTIONS
        self.likelihood_options = LIKELIHOOD_OPTIONS
        self.sampling_options = SAMPLING_OPTIONS
        self.extinction_options = EXTINCTION_OPTIONS

        self.results: list[dict[str, Any]] = []
        self.evaluation_results: list[dict[str, Any]] = []
        self.walker_history: dict[int, list[list[float]]] = {}
        self.smc_loss_offset = 0.0

        self.gen = 0
        self.num_generations = config.generations
        self.evaluation_counter = 0

        self.bounds_by_index = {
            5: (0.0, 500.0),       # amplitude
            6: (0.0, 200.0),       # baseline
            7: (-3.0, 3.0),        # trend
            8: (70.0, 230.0),      # peak_time
            9: (5.0, 90.0),        # peak_width
            10: (55.0, 180.0),     # extinction_time
            11: (2.0, 40.0),       # extinction_width
            12: (0.0, 0.98),       # extinction_depth
            13: (-3.0, 3.0),       # sampling_slope
            14: (0.1, 80.0),       # overdispersion
        }

    def get_param_bounds(self, index: int) -> tuple[float, float]:
        return self.bounds_by_index.get(index, (0.0, 1.0))

    def _reflect_at_bounds(self, value: float, lo: float, hi: float) -> float:
        return reflect_scalar(value, lo, hi)

    def _clip_categorical(self, individual: list[float]) -> None:
        individual[0] = int(np.clip(round(individual[0]), 0, len(CLADE_OPTIONS) - 1))
        individual[1] = int(np.clip(round(individual[1]), 0, len(MODEL_OPTIONS) - 1))
        individual[2] = int(np.clip(round(individual[2]), 0, len(LIKELIHOOD_OPTIONS) - 1))
        individual[3] = int(np.clip(round(individual[3]), 0, len(SAMPLING_OPTIONS) - 1))
        individual[4] = int(np.clip(round(individual[4]), 0, len(EXTINCTION_OPTIONS) - 1))

    def _repair_individual(self, individual: list[float]) -> None:
        self._clip_categorical(individual)

        for i in self.continuous_indices:
            lo, hi = self.get_param_bounds(i)
            individual[i] = self._reflect_at_bounds(float(individual[i]), lo, hi)

        # Keep extinction width positive and useful.
        individual[11] = max(2.0, individual[11])

        # If fixed K-Pg option, lock extinction time near 66 Ma.
        extinction_mode = EXTINCTION_OPTIONS[int(individual[4])]
        if extinction_mode == "kt_fixed":
            individual[10] = 66.0

    def observed_counts_for_individual(self, individual: list[float]) -> np.ndarray:
        clade = CLADE_OPTIONS[int(individual[0])]
        y = self.clade_counts.get(clade, self.raw_counts).astype(float)

        sampling = SAMPLING_OPTIONS[int(individual[3])]
        if sampling == "sqrt_corrected":
            return np.sqrt(y)
        if sampling == "log_corrected":
            return np.log1p(y)
        return y

    def predict_counts(self, individual: list[float]) -> np.ndarray:
        self._repair_individual(individual)

        model_name = MODEL_OPTIONS[int(individual[1])]
        extinction_mode = EXTINCTION_OPTIONS[int(individual[4])]

        amp = float(individual[5])
        baseline = float(individual[6])
        trend = float(individual[7])
        peak_time = float(individual[8])
        peak_width = float(individual[9])
        extinction_time = float(individual[10])
        extinction_width = float(individual[11])
        extinction_depth = float(individual[12])
        sampling_slope = float(individual[13])

        t = self.time_ma
        t_norm = (t - t.mean()) / (np.ptp(t) + 1e-12)

        if model_name == "gaussian_pulse":
            diversity = np.exp(-0.5 * ((t - peak_time) / peak_width) ** 2)

        elif model_name == "double_pulse":
            peak2 = np.clip(peak_time - 55.0, 55.0, 230.0)
            width2 = max(6.0, 0.65 * peak_width)
            diversity = (
                0.65 * np.exp(-0.5 * ((t - peak_time) / peak_width) ** 2)
                + 0.35 * np.exp(-0.5 * ((t - peak2) / width2) ** 2)
            )

        elif model_name == "logistic_decline":
            # Higher values at older times, declining toward younger bins.
            diversity = 1.0 / (1.0 + np.exp(-(t - peak_time) / max(peak_width, 1e-6)))

        else:
            raise ValueError(f"Unknown model: {model_name}")

        trend_factor = np.exp(trend * t_norm)
        sampling_bias = np.exp(sampling_slope * t_norm)

        if extinction_mode == "none":
            extinction = np.ones_like(t)
        else:
            extinction = 1.0 - extinction_depth * np.exp(
                -0.5 * ((t - extinction_time) / extinction_width) ** 2
            )
            extinction = np.clip(extinction, 1e-6, None)

        pred = baseline + amp * diversity * trend_factor * sampling_bias * extinction
        pred = np.clip(pred, 1e-9, None)

        sampling = SAMPLING_OPTIONS[int(individual[3])]
        if sampling == "sqrt_corrected":
            pred = np.sqrt(pred)
        elif sampling == "log_corrected":
            pred = np.log1p(pred)

        return np.clip(pred, 1e-9, None)

    def loss(self, individual: list[float]) -> float:
        self._repair_individual(individual)
        y = self.observed_counts_for_individual(individual)
        mu = self.predict_counts(individual)

        likelihood = LIKELIHOOD_OPTIONS[int(individual[2])]
        overdisp = max(float(individual[14]), 1e-6)

        if likelihood == "poisson":
            # Drop constants log(y!) because only relative fitness matters.
            loss = np.sum(mu - y * np.log(mu + 1e-12))

        elif likelihood == "negative_binomial":
            # NB2-style variance: var = mu + mu^2 / k.
            k = overdisp
            p = k / (k + mu)
            loss = -np.sum(
                y * np.log1p(-p + 1e-12)
                + k * np.log(p + 1e-12)
            )

        elif likelihood == "chi2":
            sigma = np.sqrt(np.maximum(y, 1.0))
            loss = np.mean(((y - mu) / sigma) ** 2)

        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")

        # Mild complexity penalties so silly huge models are discouraged.
        extinction_mode = EXTINCTION_OPTIONS[int(individual[4])]
        if extinction_mode != "none":
            loss += 0.5

        if MODEL_OPTIONS[int(individual[1])] == "double_pulse":
            loss += 0.5

        if not np.isfinite(loss):
            return 1e12

        return float(loss)

    def evaluate(self, individual: list[float]) -> tuple[tuple[float], dict[str, Any]]:
        self._repair_individual(individual)

        loss = self.loss(individual)
        observed = self.observed_counts_for_individual(individual)
        predicted = self.predict_counts(individual)

        result = {
            "individual": list(individual),
            "fitness": float(loss),
            "time_ma": self.time_ma.copy(),
            "observed": observed.copy(),
            "predicted": predicted.copy(),
            "clade": CLADE_OPTIONS[int(individual[0])],
            "model": MODEL_OPTIONS[int(individual[1])],
            "likelihood": LIKELIHOOD_OPTIONS[int(individual[2])],
            "sampling": SAMPLING_OPTIONS[int(individual[3])],
            "extinction": EXTINCTION_OPTIONS[int(individual[4])],
        }

        return (float(loss),), result

    def _record_evaluation_result(self, result: dict[str, Any]) -> None:
        result["generation"] = self.gen
        result["evaluation"] = self.evaluation_counter
        self.evaluation_counter += 1
        self.evaluation_results.append(result)

    def init_population(self, popsize: int) -> tuple[list[Any], base.Toolbox]:
        if not hasattr(creator, "FitnessMinDino"):
            creator.create("FitnessMinDino", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "DinoIndividual"):
            creator.create("DinoIndividual", list, fitness=creator.FitnessMinDino)

        toolbox = base.Toolbox()

        toolbox.register("clade_attr", lambda: random.randint(0, len(CLADE_OPTIONS) - 1))
        toolbox.register("model_attr", lambda: random.randint(0, len(MODEL_OPTIONS) - 1))
        toolbox.register("like_attr", lambda: random.randint(0, len(LIKELIHOOD_OPTIONS) - 1))
        toolbox.register("sampling_attr", lambda: random.randint(0, len(SAMPLING_OPTIONS) - 1))
        toolbox.register("ext_attr", lambda: random.randint(0, len(EXTINCTION_OPTIONS) - 1))

        for idx, name in zip(self.continuous_indices, CONTINUOUS_NAMES):
            lo, hi = self.get_param_bounds(idx)
            toolbox.register(f"{name}_attr", random.uniform, lo, hi)

        toolbox.register(
            "individual",
            tools.initCycle,
            creator.DinoIndividual,
            (
                toolbox.clade_attr,
                toolbox.model_attr,
                toolbox.like_attr,
                toolbox.sampling_attr,
                toolbox.ext_attr,
                toolbox.amplitude_attr,
                toolbox.baseline_attr,
                toolbox.trend_attr,
                toolbox.peak_time_attr,
                toolbox.peak_width_attr,
                toolbox.extinction_time_attr,
                toolbox.extinction_width_attr,
                toolbox.extinction_depth_attr,
                toolbox.sampling_slope_attr,
                toolbox.overdispersion_attr,
            ),
            n=1,
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.crossover, max_bias=0.55)
        toolbox.register("mutate", self.gaussian_mutate, base_sigma_scale=0.04)
        toolbox.register("select", self.sel_tournament, tournsize=3)

        return toolbox.population(n=popsize), toolbox

    def sel_tournament(self, individuals: list[Any], k: int | None = None, tournsize: int = 3) -> list[Any]:
        return tournament_select(individuals, k=k, tournsize=tournsize, minimize=True)

    def get_fitness_scale(self, individual: list[float]) -> float:
        if not individual.fitness.valid:
            return 1.0

        recent = [
            r["fitness"]
            for r in self.evaluation_results[-500:]
            if np.isfinite(r.get("fitness", np.inf))
        ]

        return fitness_scale(individual.fitness.values[0], recent, minimize=True)

    def crossover(self, ind1: list[float], ind2: list[float], max_bias: float = 0.55):
        f1 = ind1.fitness.values[0] if ind1.fitness.valid else float("inf")
        f2 = ind2.fitness.values[0] if ind2.fitness.valid else float("inf")

        if np.isfinite(f1) and np.isfinite(f2) and (f1 + f2) > 0:
            w1 = f2 / (f1 + f2)
            w2 = f1 / (f1 + f2)
        else:
            w1 = w2 = 0.5

        w1 = min(max_bias, max(1.0 - max_bias, w1))
        w2 = 1.0 - w1

        child1 = creator.DinoIndividual(ind1[:])
        child2 = creator.DinoIndividual(ind2[:])

        for i in self.categorical_indices:
            if random.random() < w1:
                child1[i] = ind1[i]
                child2[i] = ind2[i]
            else:
                child1[i] = ind2[i]
                child2[i] = ind1[i]

        for i in self.continuous_indices:
            noise_scale = 0.02 * abs(float(ind1[i]) - float(ind2[i])) + 1e-10
            child1[i] = w1 * ind1[i] + w2 * ind2[i] + random.gauss(0.0, noise_scale)
            child2[i] = w2 * ind1[i] + w1 * ind2[i] + random.gauss(0.0, noise_scale)

            lo, hi = self.get_param_bounds(i)
            child1[i] = self._reflect_at_bounds(child1[i], lo, hi)
            child2[i] = self._reflect_at_bounds(child2[i], lo, hi)

        return child1, child2

    def gaussian_mutate(self, individual: list[float], base_sigma_scale: float = 0.04, indpb: float = 0.3):
        fitness_scale = self.get_fitness_scale(individual)

        for i in range(len(individual)):
            if random.random() >= indpb:
                continue

            if i in self.categorical_indices:
                if random.random() < 0.10:
                    if i == 0:
                        individual[i] = random.randint(0, len(CLADE_OPTIONS) - 1)
                    elif i == 1:
                        individual[i] = random.randint(0, len(MODEL_OPTIONS) - 1)
                    elif i == 2:
                        individual[i] = random.randint(0, len(LIKELIHOOD_OPTIONS) - 1)
                    elif i == 3:
                        individual[i] = random.randint(0, len(SAMPLING_OPTIONS) - 1)
                    elif i == 4:
                        individual[i] = random.randint(0, len(EXTINCTION_OPTIONS) - 1)
                continue

            lo, hi = self.get_param_bounds(i)
            progress = self.gen / max(1, self.num_generations)
            scale = base_sigma_scale * (1.0 - 0.50 * progress)
            scale = max(0.30 * base_sigma_scale, scale)

            sigma = (hi - lo) * scale * (0.5 + 0.5 * random.random()) * fitness_scale
            individual[i] = self._reflect_at_bounds(individual[i] + random.gauss(0.0, sigma), lo, hi)

        self._repair_individual(individual)
        return individual,

    def prevent_duplicates(self, population: list[Any], toolbox: base.Toolbox) -> list[Any]:
        return deduplicate_population(
            population,
            continuous_indices=self.continuous_indices,
            get_bounds=self.get_param_bounds,
            clone=toolbox.clone,
        )

    def apply_demc_hybrid_moves(self, population: list[Any], toolbox: base.Toolbox) -> None:
        n_walkers = len(population)
        n_update = max(1, int(n_walkers * self.config.demc_fraction))

        ranked = sorted(
            range(n_walkers),
            key=lambda i: population[i].fitness.values[0] if population[i].fitness.valid else float("inf"),
            reverse=True,
        )
        update_indices = ranked[:n_update]

        d = len(self.continuous_indices)
        gamma = 2.38 / np.sqrt(2.0 * d)

        if self.gen % 6 == 0:
            gamma = 1.0

        accepted = 0

        for idx in update_indices:
            others = [i for i in range(n_walkers) if i != idx]
            if len(others) < 2:
                continue

            r1, r2 = random.sample(others, 2)

            current = population[idx]
            proposal = toolbox.clone(current)

            for i in self.continuous_indices:
                diff = population[r1][i] - population[r2][i]
                proposal[i] = current[i] + gamma * diff + random.gauss(0.0, 1e-6)

                lo, hi = self.get_param_bounds(i)
                proposal[i] = self._reflect_at_bounds(proposal[i], lo, hi)

            self._repair_individual(proposal)

            if hasattr(proposal.fitness, "values"):
                del proposal.fitness.values

            fit, result = toolbox.evaluate(proposal)
            proposal.fitness.values = fit
            self._record_evaluation_result(result)

            current_loss = current.fitness.values[0]
            proposal_loss = proposal.fitness.values[0]
            log_alpha = -(proposal_loss - current_loss)

            if np.log(random.random()) < log_alpha:
                population[idx][:] = proposal[:]
                population[idx].fitness.values = proposal.fitness.values
                accepted += 1

        if n_update > 0:
            print(f"  DE-MC: {accepted}/{n_update} accepted ({100.0 * accepted / n_update:.1f}%)")

    def run_ga(self) -> list[Any]:
        population, toolbox = self.init_population(self.config.popsize)

        for i in range(len(population)):
            self.walker_history[i] = []

        elitism_k = max(1, len(population) // 16)

        for gen in range(self.config.generations):
            self.gen = gen
            print(f"\n{'=' * 18} Generation {gen}/{self.config.generations} {'=' * 18}")

            invalid = [ind for ind in population if not ind.fitness.valid]
            if invalid:
                results = [toolbox.evaluate(ind) for ind in invalid]
                for ind, (fit, result) in zip(invalid, results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)

            best = min(population, key=lambda x: x.fitness.values[0])
            print(f"Best fitness: {best.fitness.values[0]:.6f} | {self.describe_individual(best)}")

            elites = [toolbox.clone(e) for e in tools.selBest(population, elitism_k)]

            mating_pool = toolbox.select(population)
            mating_pool = list(map(toolbox.clone, mating_pool))
            needed_children = len(population) - elitism_k
            offspring = mating_pool[:needed_children]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, "values"):
                        del c1.fitness.values
                    if hasattr(c2.fitness, "values"):
                        del c2.fitness.values

            for m in offspring:
                if random.random() < 0.6:
                    toolbox.mutate(m)
                    if hasattr(m.fitness, "values"):
                        del m.fitness.values

            offspring = self.prevent_duplicates(offspring, toolbox)

            if len(offspring) > needed_children:
                offspring = offspring[:needed_children]
            elif len(offspring) < needed_children:
                fillers = tools.selBest(population, needed_children - len(offspring))
                offspring += [toolbox.clone(f) for f in fillers]

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            if invalid:
                results = [toolbox.evaluate(ind) for ind in invalid]
                for ind, (fit, result) in zip(invalid, results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)

            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))

            population[:] = elites + offspring

            self.apply_demc_hybrid_moves(population, toolbox)

            if gen < self.config.exploration_steps:
                try:
                    moved = voronoi_explore_dearths(self, population, exploration_fraction=0.10)
                    if moved:
                        print(f"  Voronoi exploration: moved {moved} individuals")
                except Exception as exc:
                    print(f"  Voronoi exploration skipped: {exc}")

            # Voronoi exploration intentionally invalidates moved individuals.
            # Re-evaluate them before checkpoint/output selection.
            invalid = [ind for ind in population if not ind.fitness.valid]
            if invalid:
                results = [toolbox.evaluate(ind) for ind in invalid]
                for ind, (fit, result) in zip(invalid, results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)

            if gen % self.config.output_interval == 0 or gen == self.config.generations - 1:
                self.save_outputs(population, prefix=f"gen{gen:04d}_")

        self.save_outputs(population, prefix="final_")
        self.run_smc_refinement(population)

        return population

    def continuous_matrix_and_metadata(self, population: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([[float(ind[i]) for i in self.continuous_indices] for ind in population], dtype=float)
        M = np.array([[int(round(ind[i])) for i in self.categorical_indices] for ind in population], dtype=object)
        return X, M

    def individual_from_continuous_and_meta(self, theta: np.ndarray, meta: np.ndarray) -> list[float]:
        ind = [0.0] * 15
        for i, value in enumerate(meta):
            ind[i] = int(value)
        for k, idx in enumerate(self.continuous_indices):
            ind[idx] = float(theta[k])
        self._repair_individual(ind)
        return ind

    def smc_loss(self, theta: np.ndarray, meta: np.ndarray | None) -> float:
        if meta is None:
            meta = np.zeros(5, dtype=object)
        ind = self.individual_from_continuous_and_meta(theta, meta)

        # The GA objective can be negative because constant Poisson terms are
        # dropped. SMC tempering needs a stable relative non-negative loss.
        return max(0.0, self.loss(ind) - self.smc_loss_offset)

    def run_smc_refinement(self, population: list[Any]) -> None:
        print("\nRunning SMC-DEMC refinement")

        X0, metadata0 = self.continuous_matrix_and_metadata(population)
        bounds = [Bound(*self.get_param_bounds(i)) for i in self.continuous_indices]

        valid_losses = [
            float(ind.fitness.values[0])
            for ind in population
            if ind.fitness.valid and len(ind.fitness.values) > 0 and np.isfinite(ind.fitness.values[0])
        ]
        self.smc_loss_offset = min(valid_losses) if valid_losses else 0.0
        print(f"SMC relative-loss offset: {self.smc_loss_offset:.6f}")

        valid_losses = [
            float(ind.fitness.values[0])
            for ind in population
            if ind.fitness.valid and len(ind.fitness.values) > 0 and np.isfinite(ind.fitness.values[0])
        ]
        self.smc_loss_offset = min(valid_losses) if valid_losses else 0.0

        ensemble, chains_df = run_smc_demc(
            X0,
            self.smc_loss,
            bounds,
            metadata0=metadata0,
            ess_trigger=0.60,
            moves_per_stage=self.config.demc_moves_per_gen,
            rng=np.random.default_rng(self.config.seed + 1000),
            gamma_schedule=(None, 1.0),
            big_step_every=6,
            max_workers=1,
        )

        chains_path = self.config.output / "chains.csv"
        chains_df.to_csv(chains_path, index=False)

        samples = []
        for _, row in chains_df.iterrows():
            meta = np.array([row[f"m{j}"] for j in range(5)], dtype=object)
            theta = np.array([row[f"p{j}"] for j in range(10)], dtype=float)
            ind = self.individual_from_continuous_and_meta(theta, meta)
            d = self.individual_to_row(ind)
            d["fitness"] = self.loss(ind)
            samples.append(d)

        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(self.config.output / "smc_demc_samples.csv", index=False)
        samples_df.to_csv(self.config.output / "posterior_samples.csv", index=False)

        self.plot_smc_corner(samples_df)

        print(f"Saved SMC-DEMC chains: {chains_path}")
        print(f"Saved SMC-DEMC samples: {self.config.output / 'smc_demc_samples.csv'}")

    def describe_individual(self, ind: list[float]) -> str:
        return (
            f"{CLADE_OPTIONS[int(ind[0])]}, "
            f"{MODEL_OPTIONS[int(ind[1])]}, "
            f"{LIKELIHOOD_OPTIONS[int(ind[2])]}, "
            f"{SAMPLING_OPTIONS[int(ind[3])]}, "
            f"{EXTINCTION_OPTIONS[int(ind[4])]}"
        )

    def individual_to_row(self, ind: list[float]) -> dict[str, Any]:
        self._repair_individual(ind)
        row = {name: int(ind[i]) for i, name in enumerate(CATEGORICAL_NAMES)}
        for i, name in zip(self.continuous_indices, CONTINUOUS_NAMES):
            row[name] = float(ind[i])

        row["clade"] = CLADE_OPTIONS[int(ind[0])]
        row["model"] = MODEL_OPTIONS[int(ind[1])]
        row["likelihood"] = LIKELIHOOD_OPTIONS[int(ind[2])]
        row["sampling"] = SAMPLING_OPTIONS[int(ind[3])]
        row["extinction"] = EXTINCTION_OPTIONS[int(ind[4])]
        return row

    def build_results_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.evaluation_results:
            if "individual" not in r:
                continue
            row = self.individual_to_row(r["individual"])
            row["generation"] = r.get("generation", -1)
            row["evaluation"] = r.get("evaluation", -1)
            row["fitness"] = r.get("fitness", np.inf)
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("fitness", ascending=True)
            df = df.drop_duplicates(subset=PARAM_COLUMNS, keep="first")
        return df

    def save_outputs(self, population: list[Any], prefix: str = "") -> None:
        self.config.output.mkdir(parents=True, exist_ok=True)
        (self.config.output / "plots").mkdir(exist_ok=True)

        df = self.build_results_dataframe()

        if prefix == "final_":
            df.to_csv(self.config.output / "simulation_results.csv", index=False)
            df.to_csv(self.config.output / "ga_population_samples.csv", index=False)
            self.save_posterior_weights(df)
            self.save_walker_history()

        df.to_csv(self.config.output / f"{prefix}results.csv", index=False)

        valid_population = [
            ind for ind in population
            if ind.fitness.valid and len(ind.fitness.values) > 0
        ]
        if not valid_population:
            raise RuntimeError("No valid individuals available for output.")

        best_ind = min(valid_population, key=lambda x: x.fitness.values[0])
        best_result = self.evaluate(list(best_ind))[1]

        curve_path = self.save_curves_npz(best_result, prefix=prefix)
        self.plot_best_model(best_result, prefix=prefix)
        self.plot_loss_trace(prefix=prefix)

        metadata = {
            "data_source": PBDB_DINOSAURIA_CSV,
            "note": "Toy demo for GeneticMarkov; not a paleontology inference result.",
            "categorical_options": {
                "clade": CLADE_OPTIONS,
                "model": MODEL_OPTIONS,
                "likelihood": LIKELIHOOD_OPTIONS,
                "sampling": SAMPLING_OPTIONS,
                "extinction": EXTINCTION_OPTIONS,
            },
            "continuous_parameters": CONTINUOUS_NAMES,
        }
        with open(self.config.output / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved outputs: {self.config.output / f'{prefix}results.csv'}")
        print(f"Saved curves: {curve_path}")

    def save_curves_npz(self, best_result: dict[str, Any], prefix: str = "") -> Path:
        path = self.config.output / f"{prefix}curves.npz"
        np.savez_compressed(
            path,
            time_ma=best_result["time_ma"],
            observed=best_result["observed"],
            predicted=best_result["predicted"],
            best_individual=np.array(best_result["individual"], dtype=float),
        )
        return path

    def save_posterior_weights(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        loss = df["fitness"].to_numpy(float)
        valid = np.isfinite(loss)
        loss_shifted = loss - np.nanmin(loss[valid])

        temp = max(np.nanmedian(np.abs(loss_shifted[valid] - np.nanmedian(loss_shifted[valid]))), 1e-3)
        logw = np.where(valid, -loss_shifted / temp, -np.inf)
        logw -= np.nanmax(logw[valid])
        w = np.exp(logw)
        w /= np.sum(w)

        out = df.copy()
        out["weight"] = w
        out.to_csv(self.config.output / "posteriors.csv", index=False)

    def save_walker_history(self) -> None:
        if not self.walker_history:
            return

        walker_ids = np.array(sorted(self.walker_history.keys()))
        histories = np.empty(len(walker_ids), dtype=object)
        for i, wid in enumerate(walker_ids):
            histories[i] = list(self.walker_history[wid])

        np.savez_compressed(
            self.config.output / "walker_history.npz",
            walker_ids=walker_ids,
            histories=histories,
        )

    def plot_best_model(self, best_result: dict[str, Any], prefix: str = "") -> None:
        plot_dir = self.config.output / "plots"
        plot_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(best_result["time_ma"], best_result["observed"], where="mid", label="Observed PBDB bins")
        ax.plot(best_result["time_ma"], best_result["predicted"], lw=2, label="Best toy model")
        ax.axvline(66.0, ls="--", lw=1, label="K-Pg ~66 Ma")
        ax.invert_xaxis()
        ax.set_xlabel("Age [Ma]")
        ax.set_ylabel("Binned occurrence count / transformed count")
        ax.set_title(
            f"{best_result['clade']} | {best_result['model']} | "
            f"{best_result['likelihood']} | {best_result['extinction']}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}observed_vs_best_model.png", dpi=180)
        plt.close(fig)

    def plot_loss_trace(self, prefix: str = "") -> None:
        if not self.evaluation_results:
            return

        df = pd.DataFrame(
            {
                "evaluation": [r.get("evaluation", i) for i, r in enumerate(self.evaluation_results)],
                "fitness": [r.get("fitness", np.nan) for r in self.evaluation_results],
            }
        )
        df["best_so_far"] = df["fitness"].cummin()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["evaluation"], df["fitness"], ".", alpha=0.25, ms=2)
        ax.plot(df["evaluation"], df["best_so_far"], lw=2)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(self.config.output / "plots" / f"{prefix}loss_trace.png", dpi=180)
        plt.close(fig)

    def plot_smc_corner(self, samples_df: pd.DataFrame) -> None:
        if not HAS_CORNER:
            print("corner not installed; skipping SMC corner plot")
            return

        cols = [
            "amplitude",
            "baseline",
            "trend",
            "peak_time",
            "peak_width",
            "extinction_time",
            "extinction_width",
            "extinction_depth",
            "sampling_slope",
            "overdispersion",
        ]

        data = samples_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 20:
            return

        fig = corner.corner(data.to_numpy(float), labels=cols, show_titles=True)
        fig.savefig(self.config.output / "plots" / "smc_demc_corner.png", dpi=180)
        plt.close(fig)


def fetch_pbdb_data(data_dir: Path, force: bool = False) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "pbdb_dinosauria_occurrences.csv"

    if raw_path.exists() and not force:
        print(f"Using cached PBDB data: {raw_path}")
        return pd.read_csv(raw_path)

    print("Downloading PBDB Dinosauria occurrences")
    print(PBDB_DINOSAURIA_CSV)

    df = pd.read_csv(PBDB_DINOSAURIA_CSV)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw PBDB data: {raw_path}")
    return df


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


def bin_occurrences(df: pd.DataFrame, config: DemoConfig) -> pd.DataFrame:
    max_col = pick_column(df, ["max_ma", "max_ma_num"])
    min_col = pick_column(df, ["min_ma", "min_ma_num"])
    name_col = pick_column(df, ["accepted_name", "identified_name", "taxon_name", "name"])
    early_col = pick_column(df, ["early_interval"])
    late_col = pick_column(df, ["late_interval"])

    if max_col is None or min_col is None:
        raise RuntimeError(
            "PBDB output did not contain max_ma/min_ma columns. "
            "Check the API URL or the returned CSV columns."
        )

    work = df.copy()
    work[max_col] = pd.to_numeric(work[max_col], errors="coerce")
    work[min_col] = pd.to_numeric(work[min_col], errors="coerce")
    work = work[np.isfinite(work[max_col]) & np.isfinite(work[min_col])].copy()

    work["mid_ma"] = 0.5 * (work[max_col] + work[min_col])
    work = work[(work["mid_ma"] >= config.min_ma) & (work["mid_ma"] <= config.max_ma)].copy()

    # If the API gives explicit taxon names, use them for rough clade subsets.
    if name_col is not None:
        names = work[name_col].astype(str)
    else:
        names = pd.Series(["Dinosauria"] * len(work), index=work.index)

    bins = np.arange(config.min_ma, config.max_ma + config.bin_width, config.bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    out = pd.DataFrame({"time_ma": centers})
    out["count_Dinosauria"] = np.histogram(work["mid_ma"], bins=bins)[0].astype(float)

    clade_regex = {
        "Theropoda": "Theropoda|theropod|Tyrannosaur|Allosaur|Coelurosaur|Maniraptor",
        "Sauropodomorpha": "Sauropodomorpha|sauropod|prosauropod|Diplodoc|Brachiosaur",
        "Ornithischia": "Ornithischia|ornithischian|Hadrosaur|Ceratops|Stegosaur|Ankylosaur",
    }

    for clade, pattern in clade_regex.items():
        mask = names.str.contains(pattern, case=False, na=False, regex=True)
        sub = work.loc[mask, "mid_ma"]
        counts = np.histogram(sub, bins=bins)[0].astype(float)
        # If taxonomy string matching is sparse, fall back to all Dinosauria so the demo still runs.
        if counts.sum() < 20:
            counts = out["count_Dinosauria"].to_numpy(float)
        out[f"count_{clade}"] = counts

    out["count"] = out["count_Dinosauria"]

    binned_path = config.data_dir / "dinosauria_binned_counts.csv"
    out.to_csv(binned_path, index=False)
    print(f"Saved binned data: {binned_path}")

    return out


def make_output_dirs(config: DemoConfig) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "plots").mkdir(exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dinosauria GeneticMarkov demo.")
    parser.add_argument("--output", default="demos/dinosaurs/output")
    parser.add_argument("--data-dir", default="demos/dinosaurs/data")
    parser.add_argument("--popsize", type=int, default=96)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--bin-width", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DemoConfig(
        output=Path(args.output),
        data_dir=Path(args.data_dir),
        popsize=args.popsize,
        generations=args.generations,
        seed=args.seed,
        bin_width=args.bin_width,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)

    make_output_dirs(config)

    # Copy script into output for reproducibility.
    try:
        shutil.copy2(__file__, config.output / Path(__file__).name)
    except Exception:
        pass

    binned_path = config.data_dir / "dinosauria_binned_counts.csv"
    if binned_path.exists() and not args.force_download:
        print(f"Using cached binned data: {binned_path}")
        counts = pd.read_csv(binned_path)
    else:
        raw = fetch_pbdb_data(config.data_dir, force=args.force_download)
        counts = bin_occurrences(raw, config)

    problem = DinosaurProblem(counts, config)
    population = problem.run_ga()

    best = min(population, key=lambda ind: ind.fitness.values[0])
    print("\nDONE")
    print(f"Output directory: {config.output}")
    print(f"Best fitness: {best.fitness.values[0]:.6f}")
    print(problem.describe_individual(best))
    print("Best individual:")
    for name, value in problem.individual_to_row(best).items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

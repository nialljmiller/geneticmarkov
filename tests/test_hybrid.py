import random

import numpy as np
from deap import base, creator

from geneticmarkov.hybrid import apply_demc_hybrid_moves


def make_classes():
    if not hasattr(creator, "FitnessMinHybrid"):
        creator.create("FitnessMinHybrid", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "HybridIndividual"):
        creator.create("HybridIndividual", list, fitness=creator.FitnessMinHybrid)


def evaluate(ind):
    loss = float(np.sum(np.asarray(ind, dtype=float) ** 2))
    return (loss,), {"fitness": loss, "individual": list(ind)}


def test_apply_demc_hybrid_moves_runs_and_records():
    make_classes()

    pop = [creator.HybridIndividual([float(i), float(i + 1)]) for i in range(6)]

    for ind in pop:
        ind.fitness.values = evaluate(ind)[0]

    toolbox = base.Toolbox()
    toolbox.register("clone", lambda x: creator.HybridIndividual(x[:]))

    records = []

    accepted, attempted = apply_demc_hybrid_moves(
        pop,
        evaluate=evaluate,
        record_result=records.append,
        continuous_indices=[0, 1],
        get_bounds=lambda i: (-10.0, 10.0),
        clone=toolbox.clone,
        fraction=0.5,
        generation=1,
        rng=random.Random(1),
    )

    assert attempted == 3
    assert 0 <= accepted <= attempted
    assert len(records) == attempted
    assert all(ind.fitness.valid for ind in pop)
    assert all(-10.0 <= x <= 10.0 for ind in pop for x in ind)

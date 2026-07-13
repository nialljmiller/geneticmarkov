# GeneticMarkov

## Postdoc handover and developer guide

GeneticMarkov is an in-development Python package for hybrid Genetic Algorithm (GA), Differential Evolution Markov Chain (DEMC), and Sequential Monte Carlo DEMC (SMC-DEMC) exploration of expensive black-box scientific models.

This document is written primarily for the postdoc or developer taking over the project. It records what the repository is, how it relates to the published Milky Way bulge analysis, what currently works, what remains application-specific, and what should be done next.

> **Important:** GeneticMarkov is currently a pre-alpha extraction of working research code. It is not yet a stable public API and it is not yet a drop-in replacement for the original Galactic Chemical Evolution workflow.

---

## 1. Handover snapshot

At the time of handover:

- Repository: `nialljmiller/geneticmarkov`
- Active development branch: `refactor/generic-engine`
- Latest known handover commit: `a5bb1d2`
- Package version: `0.0.1`
- Development state: pre-alpha / extraction phase
- Test state: `15 passed`
- Python version used for the latest test run: Python 3.13.9
- Primary end-to-end example: `demos/dinosaurs/`
- Original scientific application: `nialljmiller/MDF_GCE_SMC_DEMC`
- Reproducible paper release: `v1.0.0-apj-accepted`
- Paper DOI: `10.3847/1538-4357/ae76f7`
- Archived scientific release DOI: `10.5281/zenodo.20418834`

The immediate goal is to finish separating the reusable optimizer and sampler from the original Milky Way bulge application while preserving the exact behavior that produced the published analysis.

The repository already contains useful generic components, but some modules still retain assumptions, parameter indices, names, and output conventions from the original GCE code.

---

## 2. Read this before changing anything

There are five important points for a new maintainer.

### 2.1 The original paper must be reproduced from the original repository

Use the tagged release of `MDF_GCE_SMC_DEMC` to reproduce the accepted paper.

Do **not** assume that the current GeneticMarkov branch reproduces the complete published pipeline. The generic package has extracted substantial parts of the machinery, but parity with the full original application has not yet been demonstrated end to end.

### 2.2 The active work is not on `main`

The current extraction work is on:

```bash
git checkout refactor/generic-engine
```

Before doing substantial work, branch from that branch rather than from `main`:

```bash
git checkout refactor/generic-engine
git pull
git checkout -b feature/<short-description>
```

### 2.3 Lower objective values are always better

The package follows the original DEAP minimization convention.

A scientific objective function should return a finite scalar loss:

```python
loss = objective(theta)
```

Lower values indicate a better fit. The SMC-DEMC code internally converts the loss into:

```python
loglike = -loss
```

Do not pass a log-likelihood into an argument named `loss_fn` unless its sign has been converted appropriately.

### 2.4 The GA population is not automatically a posterior sample

The GA is a global optimizer and parameter-space exploration method. Its final population is useful for finding viable regions and good models, but it should not be interpreted as a calibrated posterior by itself.

Posterior-style inference comes from the SMC-DEMC refinement or from independent MCMC analyses performed at fixed categorical model choices.

### 2.5 Preserve the scientific baseline before refactoring

The highest priority is behavioral parity, not aesthetic cleanup. Before replacing an implementation, save reference outputs and add regression tests.

Avoid simultaneously changing:

- parameter encoding
- initialization
- loss normalization
- random-number handling
- mutation scales
- selection behavior
- DEMC proposal geometry
- output schemas

A change in any of these can alter the scientific result.

---

## 3. Scientific origin

GeneticMarkov is extracted from the optimization and sampling machinery used in:

> **The Two-Infall Model Revisited: Constraints on Milky Way Bulge Assembly from >30,000 Galactic Chemical Evolution Models and Machine Learning**  
> Miller et al.  
> DOI: `10.3847/1538-4357/ae76f7`

The published application fits a mixed discrete/continuous, high-dimensional Galactic Chemical Evolution model. The forward model is expensive, the objective surface is not assumed to be smooth, and categorical model choices alter the physical prescription.

The method was developed to address several features that are awkward for ordinary gradient methods or a single conventional MCMC run:

- bounded continuous parameters
- categorical model choices
- multimodal objective surfaces
- expensive forward-model evaluations
- discontinuous or irregular losses
- strong parameter covariance
- local optima
- a need for both global exploration and posterior refinement

The published workflow uses the GA to explore globally, DEMC moves to refine locally during GA evolution, and separate SMC-DEMC or fixed-category MCMC calculations to assess posterior structure.

---

## 4. Algorithm implemented in the paper

The published configuration is the scientific baseline against which the generic package should be tested.

### 4.1 Initialization

The paper uses:

- population size: `Npop = 128`
- categorical parameters sampled from their available options
- continuous parameters initialized by Latin hypercube sampling
- bounded parameter ranges
- mixed categorical and continuous individual vectors

### 4.2 Selection and crossover

The GA uses:

- tournament selection with `k = 3`
- minimization fitness
- fitness-weighted crossover
- categorical inheritance probabilities capped at `0.75`
- weighted averaging plus stochastic perturbation for continuous parameters

### 4.3 Mutation

The published adaptive Gaussian mutation uses:

```text
sigma_mut(g) = sigma_0 * (1 - 0.75 * g / Gmax)
```

with:

- `sigma_0 = 0.02`
- `Gmax = 256`
- categorical mutation probability: `10%`

The mutation scale is applied relative to the parameter range in the application code.

### 4.4 Duplicate prevention and bounds

The workflow:

- detects duplicate or near-duplicate individuals
- perturbs duplicates in continuous dimensions
- reflects continuous proposals at bounds
- repairs categorical indices to valid integer choices

Reflection is preferred to clipping because it preserves more of the proposal displacement near a boundary.

### 4.5 Voronoi exploration

Sparse regions are identified from two-dimensional projections of the normalized parameter space. Poorly performing individuals can then be redirected toward large, under-sampled Voronoi cells.

This is an exploration heuristic. It is not a posterior transition kernel.

### 4.6 DEMC moves inside GA generations

After ordinary GA replacement, a configurable fraction of the population receives DEMC-style proposals.

The current generic defaults match the paper-level configuration in several important respects:

- fraction updated: `0.40`
- standard scale: `2.38 / sqrt(2 * ndim)`
- periodic large step: `gamma = 1`
- large-step cadence: every sixth generation
- reflected bounds
- Metropolis acceptance using the same scalar loss

### 4.7 SMC-DEMC refinement

The generic `run_smc_demc` function anneals from `beta = 0` to `beta = 1` using adaptive effective-sample-size control.

Current defaults are:

- ESS target/trigger: `0.60`
- DEMC moves per temperature stage: `3`
- large DEMC step every `6` stages
- default DEMC scale: `2.38 / sqrt(2 * ndim)`
- optional metadata associated with each particle
- threaded objective evaluation

The function returns:

```python
final_ensemble, chains_df
```

The chain table currently records:

- stage
- particle ID
- whether the particle accepted at least one move in that call
- metadata columns
- parameter columns

---

## 5. Repository layout and current responsibilities

The exact repository contents will evolve, but the following modules are currently important.

```text
geneticmarkov/
    __init__.py
    schema.py
    operators.py
    hybrid.py
    smc_demc.py
    exploration.py
    loss.py

demos/
    dinosaurs/
        README.md
        run_demo.py
        data/
        output/          generated; do not commit

tests/
    test_existing_loss.py
    test_existing_smc_demc.py
    test_hybrid.py
    test_operators.py
    test_package_imports.py
    test_schema.py

docs/
    legacy and inherited project documentation
```

### `geneticmarkov/schema.py`

This is the beginning of the new application-agnostic parameter representation.

It defines:

- `CategoricalParameter`
- `ContinuousParameter`
- `ParameterSchema`
- `log_uniform`
- `should_use_log`

A `ParameterSchema` stores categorical parameters first and continuous parameters second. It can:

- report parameter names and indices
- sample a valid individual
- repair a proposed individual
- return bounds
- convert an individual into a named dictionary

This module is new and is **not yet integrated through the entire package**.

### `geneticmarkov/operators.py`

This module contains generic GA helpers:

- scalar fitness extraction
- reflected scalar bounds
- tournament selection
- adaptive fitness-based mutation scaling
- DEAP fitness invalidation
- duplicate-population repair

These functions are comparatively generic and should remain independent of the astronomy application.

### `geneticmarkov/hybrid.py`

This module provides:

```python
apply_demc_hybrid_moves(...)
```

It applies DEMC-style proposals to a fraction of a DEAP population after normal GA operations.

The current function expects the caller to provide:

- an evaluation callable
- a result-recording callable
- continuous parameter indices
- a bounds lookup
- an optional repair function
- a clone function

The evaluation contract is:

```python
fitness_tuple, result_dict = evaluate(individual)
```

For a minimization problem:

```python
fitness_tuple == (loss,)
```

### `geneticmarkov/smc_demc.py`

This module contains:

- `Bound`
- `reflect_to_bounds`
- `effective_sample_size`
- `systematic_resample`
- `choose_next_beta`
- `de_mh_move`
- `run_smc_demc`

It can be used either:

1. as a standalone tempered SMC-DEMC sampler, or
2. as a refinement stage after a GA run.

### `geneticmarkov/exploration.py`

This contains the Voronoi sparse-region machinery.

**This module is not yet genuinely generic.** It still contains:

- `GalacticEvolutionGA` language in docstrings
- hard-coded parameter indices
- hard-coded GCE parameter names
- hard-coded two-dimensional exploration pairs
- assumptions about the original 15-element individual layout

This is one of the highest-priority refactoring targets.

### `geneticmarkov/loss.py`

This currently mixes two different responsibilities:

1. generally useful distance and loss metrics
2. application-specific MDF and age-metallicity losses

Generic metrics include:

- KS distance
- weighted RMSE
- MAE
- RMSE
- MAPE
- Huber loss
- cosine similarity
- log-cosh loss
- Earth Mover's Distance

The original paper-level MDF ensemble loss is:

```text
0.7 * WRMSE + 0.2 * (1 - cosine_similarity) + 0.1 * Huber
```

The generic metrics should eventually be separated from GCE-specific convenience functions.

### `demos/dinosaurs/`

This is currently the best complete demonstration of the package direction.

It uses a real, derived Paleobiology Database dataset and exercises:

- mixed categorical and continuous individuals
- DEAP population management
- tournament selection
- weighted crossover
- adaptive mutation
- duplicate prevention
- reflected bounds
- GA-integrated DEMC
- Voronoi exploration
- SMC-DEMC refinement
- CSV, NPZ, JSON, and plot outputs

It is a software demonstration, not a paleontological analysis.

The demo currently implements much of the problem setup manually. It is therefore also the best target for the first end-to-end integration of `ParameterSchema`.

---

## 6. Installation

Clone and enter the repository:

```bash
git clone git@github.com:nialljmiller/geneticmarkov.git
cd geneticmarkov
git checkout refactor/generic-engine
```

Create and activate an isolated environment.

For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the current development environment:

```bash
pip install -e ".[dev,plot,legacy-gce]"
```

The `legacy-gce` extra is currently useful even for generic development because `smc_demc.py` imports `pandas`, while `pandas` is not yet listed in the base dependencies.

That packaging mismatch should be fixed. Until then, a minimal alternative is:

```bash
pip install -e ".[dev,plot]"
pip install pandas
```

---

## 7. Test the handover state

Run:

```bash
pytest -q
```

The handover baseline is:

```text
15 passed
```

The latest known passing environment was:

```text
Python 3.13.9
pytest 8.4.1
```

The package metadata currently declares Python `>=3.8`, while the classifiers only list Python 3.8 through 3.11. Python 3.13 works for the present tests, but compatibility across the full supported range has not yet been established in CI.

After any refactor, run at least:

```bash
pytest -q
python demos/dinosaurs/run_demo.py --generations 5 --popsize 32
```

Do not merge an algorithmic change solely because the unit tests pass. The full demo should also complete and produce finite, sensible outputs.

---

## 8. Run the Dinosauria demonstration

Fast smoke test:

```bash
python demos/dinosaurs/run_demo.py \
    --generations 5 \
    --popsize 32
```

Longer demonstration:

```bash
python demos/dinosaurs/run_demo.py \
    --generations 80 \
    --popsize 96
```

Force a fresh Paleobiology Database download:

```bash
python demos/dinosaurs/run_demo.py \
    --force-download \
    --generations 80 \
    --popsize 96
```

Generated outputs are written under:

```text
demos/dinosaurs/output/
```

A successful run should produce files similar to:

```text
simulation_results.csv
ga_population_samples.csv
posteriors.csv
chains.csv
smc_demc_samples.csv
posterior_samples.csv
walker_history.npz
final_results.csv
final_curves.npz
metadata.json
```

Generation snapshots may include:

```text
gen0000_results.csv
gen0010_results.csv
...
```

Diagnostic plots are written under:

```text
demos/dinosaurs/output/plots/
```

Important plots include:

```text
final_observed_vs_best_model.png
final_loss_trace.png
smc_demc_corner.png
```

The output directory is generated and should not be committed.

---

## 9. Parameter-schema example

The new schema objects are intended to replace hand-maintained index lists.

```python
from geneticmarkov.schema import (
    CategoricalParameter,
    ContinuousParameter,
    ParameterSchema,
)

schema = ParameterSchema(
    categorical=[
        CategoricalParameter(
            name="model",
            options=["single_component", "two_component"],
        ),
        CategoricalParameter(
            name="likelihood",
            options=["gaussian", "poisson"],
        ),
    ],
    continuous=[
        ContinuousParameter("amplitude", 0.0, 100.0),
        ContinuousParameter("timescale", 0.01, 10.0, log=True),
        ContinuousParameter("offset", -5.0, 5.0),
    ],
)

individual = schema.sample()
individual = schema.repair(individual)
named_parameters = schema.as_dict(individual)

print(schema.names)
print(schema.categorical_indices)
print(schema.continuous_indices)
print(named_parameters)
```

Categorical values are encoded internally as integer indices, represented in the mixed numerical vector as floats. `as_dict` provides both the selected option and the corresponding index:

```python
{
    "model": "two_component",
    "model_idx": 1,
    ...
}
```

The next maintainer should use `ParameterSchema` as the single source of truth for:

- names
- categorical options
- bounds
- parameter ordering
- repair
- serialization

---

## 10. Standalone SMC-DEMC example

```python
import numpy as np

from geneticmarkov.smc_demc import Bound, run_smc_demc


rng = np.random.default_rng(42)

bounds = [
    Bound(-5.0, 5.0),
    Bound(-5.0, 5.0),
]

initial_ensemble = rng.uniform(
    low=[-5.0, -5.0],
    high=[5.0, 5.0],
    size=(128, 2),
)


def loss_fn(theta, metadata):
    theta = np.asarray(theta, dtype=float)
    return float(np.sum(theta**2))


final_ensemble, chain = run_smc_demc(
    X0=initial_ensemble,
    loss_fn=loss_fn,
    bounds=bounds,
    ess_trigger=0.60,
    moves_per_stage=3,
    rng=rng,
    big_step_every=6,
)

print(final_ensemble.shape)
print(chain.head())
```

Important conventions:

- `loss_fn` returns a scalar loss, not a log-likelihood.
- lower is better
- every loss must be finite
- the ensemble must contain at least three particles for differential-evolution proposals
- realistic use requires substantially more than three particles
- bounds are reflected
- `metadata` is passed through to the loss function without being modified by continuous DEMC moves

---

## 11. Adapting GeneticMarkov to a new scientific problem

A new application should define a problem layer rather than modify the generic sampler.

The problem layer should own:

1. observed-data loading
2. parameter schema
3. forward-model execution
4. prediction extraction
5. loss computation
6. invalid-model handling
7. result serialization
8. any application-specific plots

A useful conceptual interface is:

```python
class ScientificProblem:
    schema: ParameterSchema

    def predict(self, parameters: dict) -> object:
        ...

    def loss(self, prediction: object) -> float:
        ...

    def evaluate(self, individual):
        parameters = self.schema.as_dict(individual)
        prediction = self.predict(parameters)
        loss = self.loss(prediction)

        return (loss,), {
            "loss": loss,
            "parameters": parameters,
            "prediction": prediction,
        }
```

The current Dinosauria `DinosaurProblem` class is the best working template, but it manually stores index maps and categorical lists. Replacing those duplicated structures with `ParameterSchema` is a sensible first integration task.

When wrapping a scientific model:

- return a large finite penalty for failed simulations
- do not return `NaN`
- record the failure reason separately
- keep units explicit
- record software versions and configuration
- separate model execution from objective calculation
- keep raw model outputs linked to parameter rows
- make random seeds explicit

---

## 12. Reproducing the published bulge analysis

The exact paper analysis belongs to the original repository:

```text
nialljmiller/MDF_GCE_SMC_DEMC
```

Use:

```text
v1.0.0-apj-accepted
```

and the archived release:

```text
10.5281/zenodo.20418834
```

The accepted paper uses a 15-dimensional mixed parameter space and more than 30,000 GCE models. The analysis combines:

- the bulge metallicity distribution function
- alpha-element abundance trends
- age-metallicity information
- categorical physical prescriptions
- continuous infall and star-formation parameters
- GA exploration
- hybrid DEMC moves
- independent fixed-category MCMC checks

The accepted analysis also combines independent MCMC calculations over categorical configurations to validate the GA+DEMC result. This is important: a generic continuous sampler with categorical labels carried as immutable metadata is not by itself equivalent to a fully trans-dimensional Bayesian sampler.

### Parity benchmark required before replacement

Before declaring that GeneticMarkov replaces the original machinery, construct a parity benchmark that holds constant:

- input data
- model code
- parameter bounds
- categorical options
- parameter order
- initial population
- all random seeds
- objective function
- number of generations
- population size
- selection settings
- crossover settings
- mutation settings
- DEMC settings
- Voronoi settings

Compare:

- generation-zero population
- per-generation best loss
- loss distribution by generation
- accepted DEMC counts
- duplicate-repair counts
- categorical frequencies
- final ranked models
- final continuous parameter distributions
- output-table schemas
- posterior summary statistics

Exact trajectories may differ after parallel evaluations or changes in random-number ordering, but statistical agreement must be demonstrated and documented.

---

## 13. Known technical debt and sharp edges

This section should be kept current. It is more useful to the replacement postdoc than a generic roadmap.

### 13.1 Base dependency mismatch

`geneticmarkov.smc_demc` imports `pandas`, but `pandas` is not currently in the base dependency list.

This means importing the top-level package can fail in a nominal minimal installation because `__init__.py` imports `smc_demc`.

Recommended fix:

- add `pandas` to base dependencies, or
- remove the hard pandas requirement from the core sampler and make DataFrame conversion optional

### 13.2 The public API is incomplete

The top-level package currently exports:

- `Bound`
- `reflect_to_bounds`
- `run_smc_demc`

It does not yet export:

- schema classes
- hybrid helpers
- generic operators
- exploration helpers

Do not expand `__all__` casually. First decide which interfaces are intended to remain stable.

### 13.3 `exploration.py` remains GCE-specific

The Voronoi module contains hard-coded indices and GCE parameter names.

The generic version should accept exploration pairs explicitly, for example:

```python
exploration_pairs = [
    ("timescale_1", "timescale_2"),
    ("amplitude", "width"),
]
```

Those names should be resolved through `ParameterSchema`, not integer literals.

### 13.4 `loss.py` mixes generic and scientific code

Split it into something like:

```text
geneticmarkov/metrics.py
geneticmarkov/losses.py
examples/gce/loss.py
```

Do not delete the original functions until regression tests verify identical numerical behavior.

### 13.5 Random-number handling is mixed

The code currently uses both:

- Python's `random`
- NumPy generators

For reproducibility, a high-level runner should seed both explicitly and pass RNG objects down into every stochastic component.

Avoid hidden global RNG use.

### 13.6 Threaded evaluation is not a universal parallel solution

`de_mh_move` currently uses a `ThreadPool`.

Threads may help when the forward model releases the GIL or spends most of its time in compiled code or external processes. They will not efficiently parallelize all Python-bound objective functions.

A future backend should support:

- serial execution
- threads
- processes
- scheduler/HPC execution

The forward model may also be non-thread-safe.

### 13.7 SMC diagnostics are printed but incompletely persisted

The SMC-DEMC run prints:

- stage
- beta
- ESS
- acceptance fraction

The current returned chain table does not persist all of those stage-level diagnostics.

Add a stage diagnostics table or include:

- beta
- ESS
- acceptance rate
- resampling flag
- gamma
- wall time
- evaluation count

### 13.8 Large-step behavior at generation zero should be reviewed

`apply_demc_hybrid_moves` currently sets `gamma = 1` whenever:

```python
generation % big_step_every == 0
```

This includes generation zero.

Decide explicitly whether generation zero should count as a scheduled large-step generation. Preserve the published behavior in the parity path even if the generic default is changed later.

### 13.9 Numerical stability needs explicit testing

SMC weights use exponentials of loss differences. Very large losses can underflow.

Add tests using:

- large positive losses
- nearly identical losses
- pathological particles
- all-invalid populations
- narrow and wide posteriors

Use numerically stable shifted log-weight calculations where appropriate.

### 13.10 Categorical inference is not trans-dimensional MCMC

The present mixed-vector GA can explore categorical choices, but continuous DEMC moves do not change categorical metadata.

The paper addresses this through separate calculations at fixed categorical choices. A fully Bayesian treatment of changing model dimension or model choice would require a method such as reversible-jump MCMC or explicit marginalization.

Do not describe the current categorical handling as a general trans-dimensional posterior sampler.

---

## 14. Recommended development priorities

### Priority 0: preserve the baseline

Before broad refactoring:

- archive current demo outputs
- add fixed-seed regression fixtures
- record the active branch and commit
- verify the 15-test baseline
- verify the smoke demo
- document known numerical outputs

### Priority 1: complete the parameter abstraction

Integrate `ParameterSchema` into the Dinosauria demo and remove duplicated:

- name lists
- index lists
- bounds dictionaries
- categorical option lists
- repair logic

Then use the same interface in a small synthetic benchmark.

### Priority 2: remove GCE assumptions from the generic core

Refactor `exploration.py` so that it receives:

- continuous parameter indices or names
- exploration pairs
- bounds
- repair callbacks
- RNG objects

Move original GCE exploration-pair definitions back into the GCE application.

### Priority 3: define the user-facing runner

The project needs a single high-level entry point.

A possible direction is:

```python
sampler = GeneticMarkovSampler(
    schema=schema,
    objective=objective,
    population_size=128,
    seed=42,
)

result = sampler.run(
    generations=256,
    demc_fraction=0.40,
    smc_refine=True,
)
```

Do not implement this until the lower-level contracts are stable.

### Priority 4: formalize result objects and outputs

Define a result container containing:

- run configuration
- parameter schema
- generation history
- evaluation history
- final population
- best model
- SMC samples
- stage diagnostics
- random seeds
- software versions
- output paths

The output schema should be versioned.

### Priority 5: checkpoint and resume

Expensive scientific models require reliable restart behavior.

A checkpoint should preserve:

- current generation
- population
- fitnesses
- RNG state
- evaluation counter
- result history
- DEMC state
- configuration
- schema
- software version

### Priority 6: validation suite

Add known-target benchmarks:

- one-dimensional Gaussian
- correlated multivariate Gaussian
- bounded Gaussian near a boundary
- bimodal Gaussian mixture
- mixed categorical model selection
- Rosenbrock-type objective
- failed-model penalty behavior

Compare posterior moments and credible intervals against analytical or trusted reference results.

### Priority 7: CI and packaging

Before a public release:

- test supported Python versions
- fix dependency declarations
- add lint/type checks
- add build checks
- build wheel and source distribution
- test installation into a clean environment
- decide the minimum supported Python version
- add changelog and contribution guidance

---

## 15. What not to do

Do not:

- treat the final GA population as a calibrated posterior
- change parameter order without a migration layer
- hard-code new scientific parameter names into the generic core
- silently change objective normalization
- silently convert a loss into a likelihood with the wrong sign
- clip proposals when parity requires reflection
- change random-number ordering and then compare trajectories as if they were identical
- assume threaded execution is process-safe
- delete the accepted paper tag
- replace the original scientific repository before parity is demonstrated
- publish a stable API while the schema and runner are still changing

---

## 16. Suggested first day for the replacement postdoc

Run the following:

```bash
git clone git@github.com:nialljmiller/geneticmarkov.git
cd geneticmarkov
git checkout refactor/generic-engine

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,plot,legacy-gce]"

pytest -q

python demos/dinosaurs/run_demo.py \
    --generations 5 \
    --popsize 32
```

Then read, in this order:

1. this handover document
2. `demos/dinosaurs/README.md`
3. `demos/dinosaurs/run_demo.py`
4. `geneticmarkov/schema.py`
5. `geneticmarkov/operators.py`
6. `geneticmarkov/hybrid.py`
7. `geneticmarkov/smc_demc.py`
8. `geneticmarkov/exploration.py`
9. `geneticmarkov/loss.py`
10. the paper methodology and validation appendix
11. the original `MDF_GCE_SMC_DEMC` tagged release

The first concrete pull request should be small. A good candidate is:

- fix the pandas dependency issue
- add CI for the existing tests
- preserve all behavior

The first algorithmic pull request should add or extend regression tests before changing implementation.

---

## 17. Handover completion checklist

Before the original developer leaves, the incoming maintainer should be able to:

- clone the private repository
- access the active branch
- install the development environment
- run all tests
- run the smoke demonstration
- locate the original paper code and accepted tag
- explain the difference between GA exploration and posterior sampling
- identify which modules are generic
- identify which modules remain GCE-specific
- reproduce at least one fixed-seed benchmark
- understand the output files
- create a feature branch and pull request
- know where scientific data and large outputs are stored
- know which outputs must not be committed

Items that still require project-specific confirmation should be recorded in an issue rather than left only in private messages.

---

## 18. Roadmap toward a JOSS-ready package

A JOSS submission should wait until the software has a coherent, documented, testable public workflow.

Minimum requirements:

- stable package name and scope
- clear public API
- generic parameter schema used end to end
- at least two scientifically distinct examples
- documented installation
- documented outputs
- automated test suite
- CI across supported Python versions
- clean dependency declarations
- archived release
- contribution guide
- code of conduct
- software paper describing the generic package rather than only the bulge application
- explicit comparison with established optimization and ensemble-MCMC tools
- clear explanation of where GeneticMarkov is appropriate and where it is not

The strongest scientific claim is not that this replaces ordinary MCMC. It is that the package provides a practical workflow for difficult mixed categorical/continuous black-box problems where global search is required before local posterior refinement becomes reliable.

---

## 19. Citation

For the scientific method and original Galactic Chemical Evolution application, cite:

```bibtex
@article{Miller2026TwoInfall,
  title   = {The Two-Infall Model Revisited: Constraints on Milky Way Bulge Assembly from >30,000 Galactic Chemical Evolution Models and Machine Learning},
  author  = {Miller, Niall and collaborators},
  journal = {The Astrophysical Journal},
  year    = {2026},
  doi     = {10.3847/1538-4357/ae76f7}
}
```

For the archived accepted scientific code:

```text
https://doi.org/10.5281/zenodo.20418834
```

Update this section when GeneticMarkov receives its own archived release and software citation.

---

## 20. License

MIT.

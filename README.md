# GeneticMarkov

GeneticMarkov is an in-development Python package for hybrid Genetic Algorithm and Differential Evolution Markov Chain exploration of black-box scientific models.

The package is being extracted from a Galactic Chemical Evolution fitting codebase into a reusable sampler framework. The goal is to provide an interface that feels familiar to users of ensemble MCMC tools, while retaining the global-search behavior of a genetic algorithm and the local refinement behavior of DEMC.

## Current status

This repository is in the extraction phase.

The original GCE-specific implementation is still present under `mdf_gce/`. The generic package namespace is `geneticmarkov/`. The immediate development goal is to separate the reusable GA+DEMC machinery from the original astronomy application without replacing or simplifying the existing algorithm.

The current working method includes:

- mixed categorical and continuous parameter vectors
- DEAP-based genetic populations
- tournament selection
- fitness-weighted crossover
- adaptive Gaussian mutation
- categorical mutation
- reflected parameter bounds
- duplicate prevention
- DEMC hybrid moves inside GA generations
- Voronoi sparse-region exploration
- SMC-DEMC posterior-style refinement
- checkpoint-like generation outputs
- CSV, NPZ, chain, posterior, and plot outputs

## Intended use case

GeneticMarkov is aimed at expensive scientific fitting problems where the parameter space is awkward for ordinary MCMC:

- bounded continuous parameters
- categorical model choices
- discontinuous likelihoods
- multimodality
- expensive forward models
- mixed optimization and posterior-exploration workflows

It is not intended to replace conventional MCMC in simple well-behaved continuous problems. It is intended to complement MCMC when the first challenge is finding and characterizing viable regions of a difficult parameter space.

## Demos

- [Dinosauria fossil occurrence demo](demos/dinosaurs/README.md): a real-data, non-astronomy example fitting PBDB dinosaur fossil occurrence counts with the GA+DEMC workflow.

## Development install

From the repository root:

    pip install -e .

For development dependencies:

    pip install -e ".[dev]"

## Run tests

    pytest -q

## Repository layout

    geneticmarkov/       Generic package namespace under extraction
    mdf_gce/             Legacy GCE-specific implementation
    demos/               Demonstrations of the method outside the original application
    tests/               Regression tests for existing GA/DEMC components
    docs/                Existing project documentation from the source codebase

## Roadmap

Near-term work:

1. Preserve the current working GA+DEMC behavior with regression tests.
2. Move generic DEMC, bounds, exploration, operator, and output utilities into `geneticmarkov/`.
3. Keep the original GCE fitter as an application/example rather than the package core.
4. Build an emcee-like public API around the existing algorithm.
5. Add more demos and benchmark problems that exercise mixed categorical/continuous fitting.

## License

MIT.

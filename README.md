# GeneticMarkov

An emcee-like hybrid genetic algorithm and Differential Evolution Markov Chain sampler for expensive black-box scientific models.

This repository was bootstrapped from a Galactic Chemical Evolution fitting project. The immediate goal is to extract the generic GA+DEMC machinery into a reusable Python package with an interface that feels familiar to users of `emcee`.

## Target API

```python
import geneticmarkov as gm

sampler = gm.EnsembleSampler(nwalkers, ndim, log_prob, bounds=bounds)
sampler.run_mcmc(p0, nsteps)

samples = sampler.get_chain(flat=True)
log_prob = sampler.get_log_prob(flat=True)




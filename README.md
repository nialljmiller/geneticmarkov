# GeneticMarkov

An emcee-like hybrid genetic algorithm and Differential Evolution Markov Chain sampler for expensive black-box scientific models.

## Target API

```python
import geneticmarkov as gm

sampler = gm.EnsembleSampler(nwalkers, ndim, log_prob, bounds=bounds)
sampler.run_mcmc(p0, nsteps)

samples = sampler.get_chain(flat=True)
log_prob = sampler.get_log_prob(flat=True)
````

## Current status

Early extraction. The generic sampler will live in `geneticmarkov/`; the old GCE-specific code is still present temporarily while the algorithm is being separated.

````

Then:

```bash
git add README.md
git commit -m "Fix README bootstrap text"
git push
````

Also, do **not** use `git rm` for `mdf_gce.egg-info`; it is untracked. Just delete the local generated directories:

```bash
rm -rf mdf_gce.egg-info geneticmarkov.egg-info
```

Then add them to `.gitignore` so they stop appearing:

```bash
printf '\n*.egg-info/\n__pycache__/\n*.pyc\n' >> .gitignore
git add .gitignore
git commit -m "Ignore generated Python metadata"
git push
```

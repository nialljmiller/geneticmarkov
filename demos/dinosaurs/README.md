# Dinosauria fossil occurrence demo

This demo applies the current GeneticMarkov GA+DEMC workflow to a real non-astronomy dataset: Dinosauria fossil occurrence counts from the Paleobiology Database.

The purpose of the demo is to show that the method is not intrinsically tied to Galactic Chemical Evolution. It demonstrates the same kind of mixed categorical and continuous fitting problem in a domain that is easy to understand visually: fossil occurrences through geological time.

This is a software demonstration, not a paleontological analysis.

## What the demo does

The demo fits a simplified occurrence-count model to binned Dinosauria fossil records.

At a high level, the model is:

    predicted count = baseline + diversity curve * sampling bias * extinction factor

The optimizer searches over both model choices and numerical parameters. This makes the demo structurally similar to the original GCE fitting problem: some parameters choose model families, while others tune continuous values.

The demo then produces the same kind of outputs expected from a serious fitting workflow:

- generation-by-generation results
- final ranked parameter table
- posterior-weighted table
- SMC-DEMC chain table
- posterior samples
- walker history
- saved model curves
- diagnostic plots

## Data

The raw data source is the Paleobiology Database.

The PBDB query used by the demo is:

    https://paleobiodb.org/data1.2/occs/list.csv?base_name=Dinosauria&show=class,classext,ident,phylo,time&limit=all

The committed data file is:

    demos/dinosaurs/data/dinosauria_binned_counts.csv

This committed CSV is a derived data product. It bins Dinosauria fossil occurrences into 5 Myr bins between 50 and 250 Ma.

By default, the demo uses the committed binned file. This means the example can run offline and gives reproducible behavior. If `--force-download` is passed, the script re-downloads the raw PBDB occurrence table and rebuilds the binned data.

The raw downloaded PBDB table is not committed.

## Why this is a useful demo

A Gaussian toy problem would only show that the code can optimize a smooth function. That is not enough for this package.

The dinosaur occurrence problem is more useful because it has several features that are closer to real scientific fitting:

- the data are real
- the counts are noisy
- the observations are time-binned
- the parameter space is bounded
- there are competing model families
- some parameters are categorical
- some parameters are continuous
- the likelihood choice itself can vary
- the result is easy to plot and inspect

The demo is intentionally not astronomy-specific, so users can see the general shape of the method without knowing Galactic Chemical Evolution.

## Parameterization

Each individual has five categorical parameters followed by ten continuous parameters.

### Categorical parameters

| Parameter | Meaning |
|---|---|
| `clade_idx` | Which Dinosauria subset to fit |
| `model_idx` | Which diversity-curve family to use |
| `likelihood_idx` | Which loss/likelihood model to use |
| `sampling_idx` | Which count transformation to apply |
| `extinction_idx` | Which extinction/suppression model to use |

Current options:

| Category | Options |
|---|---|
| clade | Dinosauria, Theropoda, Sauropodomorpha, Ornithischia |
| model | gaussian_pulse, double_pulse, logistic_decline |
| likelihood | poisson, negative_binomial, chi2 |
| sampling | raw, sqrt_corrected, log_corrected |
| extinction | none, kt_fixed, free_pulse |

### Continuous parameters

| Parameter | Meaning |
|---|---|
| `amplitude` | Scale of the main occurrence curve |
| `baseline` | Background occurrence level |
| `trend` | Broad time-dependent trend |
| `peak_time` | Age of the main diversity peak |
| `peak_width` | Width of the main diversity peak |
| `extinction_time` | Age of the extinction/suppression pulse |
| `extinction_width` | Width of the extinction/suppression pulse |
| `extinction_depth` | Strength of the extinction/suppression pulse |
| `sampling_slope` | Simple preservation/sampling-bias term |
| `overdispersion` | Negative-binomial dispersion parameter |

## GeneticMarkov features exercised

This demo is intended to exercise the full current method rather than a stripped-down sampler.

It uses:

- mixed categorical and continuous individuals
- DEAP population and fitness handling
- tournament selection
- fitness-weighted crossover
- adaptive Gaussian mutation
- categorical mutation
- duplicate prevention
- reflected bounds
- DEMC hybrid moves during GA generations
- periodic large DEMC jumps
- Voronoi sparse-region exploration
- generation-level output files
- final ranked result tables
- posterior-weighted result tables
- SMC-DEMC refinement
- chain output
- posterior sample output
- walker history output
- linked curve output
- diagnostic plots

## Running the demo

From the repository root, run:

    python demos/dinosaurs/run_demo.py --generations 80 --popsize 96

For a fast smoke test, run:

    python demos/dinosaurs/run_demo.py --generations 5 --popsize 32

To force a fresh PBDB download and rebuild the binned data, run:

    python demos/dinosaurs/run_demo.py --force-download --generations 80 --popsize 96

## Expected runtime behavior

During the run, the terminal should show generation blocks like:

    Generation 0/80
    Best fitness: ...
    DE-MC: ...
    Voronoi exploration: moved ...

For early generations, Voronoi exploration should move a fraction of the population into sparse regions. DEMC hybrid moves should report acceptance counts each generation.

At the end, the script should run SMC-DEMC refinement and report stages with beta, ESS, and acceptance fraction.

## Expected output directory

By default, outputs are written to:

    demos/dinosaurs/output/

This directory is generated and should not be committed.

## Expected output files

A successful run should produce files like:

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

The run also writes generation snapshots such as:

    gen0000_results.csv
    gen0010_results.csv
    gen0020_results.csv
    gen0030_results.csv
    gen0040_results.csv
    gen0050_results.csv
    gen0060_results.csv
    gen0070_results.csv
    gen0079_results.csv

The exact generation files depend on `--generations` and the output interval.

## Expected plots

Plots are written to:

    demos/dinosaurs/output/plots/

Important plots include:

    final_observed_vs_best_model.png
    final_loss_trace.png
    smc_demc_corner.png

The observed-vs-best plot compares the binned PBDB occurrence counts against the best toy model. The loss trace shows optimizer progress. The corner plot summarizes the continuous parameters from the SMC-DEMC refinement stage.

## Interpreting the result

The default run should usually find a broad, smooth occurrence-count structure. It may prefer a simple Gaussian-like occurrence history, sometimes with no explicit extinction pulse.

That should not be interpreted as a claim about dinosaur macroevolution. The model is deliberately simple and the data are not corrected rigorously for sampling or preservation effects.

The meaningful result is that the optimizer successfully handles:

- real data
- categorical model choices
- continuous parameters
- noisy count fitting
- bounded parameters
- hybrid GA and DEMC exploration
- posterior-style refinement and outputs

## Caveats

PBDB occurrence counts are not corrected here for all known paleontological biases. In particular, this demo does not rigorously model:

- preservation bias
- collection effort
- rock availability
- taxonomic revision
- uneven sampling through time
- geographic sampling variation
- publication and database-entry bias

The clade-specific subsets are approximate. PBDB exports can vary in available taxonomic columns, so if a subgroup is too sparse the demo falls back to full Dinosauria counts to keep the software example robust.

This is a package demonstration, not a scientific interpretation of dinosaur diversity or extinction.

## Why this belongs in GeneticMarkov

This example demonstrates the intended direction of the package.

A user should be able to bring a black-box model, define mixed categorical and continuous parameters, run a GA+DEMC search, and get interpretable outputs. The Dinosauria demo shows that workflow in a compact, non-astronomy setting.

It is therefore a useful counterpart to the original GCE application: same optimization machinery, different scientific domain.

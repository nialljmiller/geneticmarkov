import random

import numpy as np
from deap import base, creator

from geneticmarkov.operators import (
    deduplicate_population,
    fitness_scale,
    reflect_scalar,
    tournament_select,
)


def test_reflect_scalar():
    assert np.isclose(reflect_scalar(-0.2, 0.0, 1.0), 0.2)
    assert np.isclose(reflect_scalar(1.3, 0.0, 1.0), 0.7)
    assert np.isclose(reflect_scalar(0.4, 0.0, 1.0), 0.4)


def test_fitness_scale_minimize():
    recent = list(range(100))
    assert fitness_scale(5, recent, minimize=True) == 0.5
    assert fitness_scale(50, recent, minimize=True) == 1.0
    assert fitness_scale(95, recent, minimize=True) == 1.5


def test_tournament_select_minimize():
    if not hasattr(creator, "FitnessMinOperators"):
        creator.create("FitnessMinOperators", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "OperatorIndividual"):
        creator.create("OperatorIndividual", list, fitness=creator.FitnessMinOperators)

    pop = [creator.OperatorIndividual([i]) for i in range(5)]
    for i, ind in enumerate(pop):
        ind.fitness.values = (float(i),)

    selected = tournament_select(pop, k=5, tournsize=3, rng=random.Random(1))

    assert len(selected) == 5
    assert all(ind.fitness.valid for ind in selected)


def test_deduplicate_population_jitters_duplicate():
    if not hasattr(creator, "FitnessMinOperators"):
        creator.create("FitnessMinOperators", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "OperatorIndividual"):
        creator.create("OperatorIndividual", list, fitness=creator.FitnessMinOperators)

    ind1 = creator.OperatorIndividual([0.5, 0.5])
    ind2 = creator.OperatorIndividual([0.5, 0.5])
    ind1.fitness.values = (1.0,)
    ind2.fitness.values = (1.0,)

    toolbox = base.Toolbox()
    toolbox.register("clone", lambda x: creator.OperatorIndividual(x[:]))

    out = deduplicate_population(
        [ind1, ind2],
        continuous_indices=[0, 1],
        get_bounds=lambda i: (0.0, 1.0),
        clone=toolbox.clone,
        rng=random.Random(2),
    )

    assert len(out) == 2
    assert out[0] == [0.5, 0.5]
    assert out[1] != [0.5, 0.5]
    assert all(0.0 <= x <= 1.0 for x in out[1])

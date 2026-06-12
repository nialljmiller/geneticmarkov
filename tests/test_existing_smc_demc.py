import numpy as np

from mdf_gce.core.smc_demc import (
    Bound,
    reflect_to_bounds,
    effective_sample_size,
    systematic_resample,
    choose_next_beta,
)


def test_reflect_to_bounds_keeps_values_inside():
    x = np.array([-0.2, 1.3, 0.5])
    bounds = [Bound(0.0, 1.0), Bound(0.0, 1.0), Bound(0.0, 1.0)]

    y = reflect_to_bounds(x, bounds)

    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)
    assert np.allclose(y, [0.2, 0.7, 0.5])


def test_effective_sample_size_uniform_weights():
    weights = np.ones(10) / 10
    assert np.isclose(effective_sample_size(weights), 10.0)


def test_systematic_resample_shape_and_range():
    rng = np.random.default_rng(1)
    weights = np.ones(8) / 8

    idx = systematic_resample(weights, rng)

    assert idx.shape == (8,)
    assert np.all(idx >= 0)
    assert np.all(idx < 8)


def test_choose_next_beta_increases_beta():
    loss = np.array([0.1, 0.2, 0.5, 1.0])
    beta_next = choose_next_beta(loss, beta_prev=0.0)

    assert 0.0 < beta_next <= 1.0

import numpy as np

from mdf_gce.core.loss import (
    compute_ks_distance,
    compute_wrmse,
    compute_mae,
    compute_rmse,
    compute_cosine_similarity,
    compute_ensemble_loss,
)


def test_identical_distributions_have_low_loss():
    observed = np.array([0.1, 0.3, 0.6])
    predicted = observed.copy()
    sigma = np.ones_like(observed)

    assert np.isclose(compute_ks_distance(observed, predicted), 0.0)
    assert np.isclose(compute_wrmse(observed, predicted, sigma), 0.0)
    assert np.isclose(compute_mae(observed, predicted), 0.0)
    assert np.isclose(compute_rmse(observed, predicted), 0.0)
    assert np.isclose(compute_cosine_similarity(observed, predicted), 1.0)
    assert compute_ensemble_loss(observed, predicted, sigma) < 1e-12

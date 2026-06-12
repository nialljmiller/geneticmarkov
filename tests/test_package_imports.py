def test_mdf_gce_imports():
    import mdf_gce
    assert mdf_gce.__version__


def test_core_imports():
    import mdf_gce.core as core
    assert core.Bound is not None
    assert core.run_smc_demc is not None
    assert core.de_mh_move is not None

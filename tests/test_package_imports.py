def test_geneticmarkov_imports():
    import geneticmarkov
    assert geneticmarkov.__version__


def test_core_imports():
    from geneticmarkov.smc_demc import Bound, run_smc_demc, de_mh_move
    assert Bound is not None
    assert run_smc_demc is not None
    assert de_mh_move is not None

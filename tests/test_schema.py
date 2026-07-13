from geneticmarkov.schema import (
    CategoricalParameter,
    ContinuousParameter,
    ParameterSchema,
    should_use_log,
)


def test_should_use_log():
    assert should_use_log(1.0, 100.0)
    assert not should_use_log(1.0, 10.0)
    assert not should_use_log(-1.0, 100.0)


def test_schema_sample_repair_and_dict():
    schema = ParameterSchema(
        categorical=[
            CategoricalParameter("model", ["a", "b"]),
            CategoricalParameter("likelihood", ["normal", "poisson"]),
        ],
        continuous=[
            ContinuousParameter("x", 0.0, 1.0),
            ContinuousParameter("scale", 1.0, 100.0, log=True),
        ],
    )

    sample = schema.sample()
    assert len(sample) == 4

    repaired = schema.repair([99, -4, -1.0, 1e9])
    assert repaired == [1.0, 0.0, 0.0, 100.0]

    d = schema.as_dict([1, 0, 0.25, 10.0])
    assert d["model"] == "b"
    assert d["model_idx"] == 1
    assert d["likelihood"] == "normal"
    assert d["x"] == 0.25
    assert d["scale"] == 10.0


def test_get_bounds():
    schema = ParameterSchema(
        categorical=[CategoricalParameter("model", ["a"])],
        continuous=[ContinuousParameter("x", -2.0, 3.0)],
    )

    assert schema.categorical_indices == [0]
    assert schema.continuous_indices == [1]
    assert schema.get_bounds(1) == (-2.0, 3.0)

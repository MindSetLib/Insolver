from insolver.FeaturesFunctions.VehicleFunctions import f_power, f_vehicle_age


def test_f_power():
    assert f_power(None) is None
    assert f_power(5) is None
    assert f_power(600) is None
    assert f_power(200.55) == 20.0


def test_f_vehicle_age():
    assert f_vehicle_age(None) is None
    assert f_vehicle_age(-10) is None
    assert f_vehicle_age(100) == 25

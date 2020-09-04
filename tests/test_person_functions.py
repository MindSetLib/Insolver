from insolver.FeaturesFunctions.PersonFunctions import f_gen_age_exp


def test_f_gen_age_exp():
    assert f_gen_age_exp(1, 2, 24, 4) == 1 * 100 + 0 * 10 + 0
    assert f_gen_age_exp(2, 1, 24, 4) == 2 * 100 + 0 * 10 + 0
    assert f_gen_age_exp(2, 2, 24, 4) == 0 * 100 + 0 * 10 + 0

    assert f_gen_age_exp(1, 2, 26, 4) == 1 * 100 + 1 * 10 + 0
    assert f_gen_age_exp(1, 2, 31, 4) == 1 * 100 + 2 * 10 + 0
    assert f_gen_age_exp(1, 2, 36, 4) == 1 * 100 + 3 * 10 + 0
    assert f_gen_age_exp(1, 2, 41, 4) == 1 * 100 + 4 * 10 + 0
    assert f_gen_age_exp(1, 2, 46, 4) == 1 * 100 + 5 * 10 + 0
    assert f_gen_age_exp(1, 2, 80, 4) == 1 * 100 + 6 * 10 + 0

    assert f_gen_age_exp(1, 2, 26, 6) == 1 * 100 + 1 * 10 + 1
    assert f_gen_age_exp(1, 2, 26, 11) == 1 * 100 + 1 * 10 + 2
    assert f_gen_age_exp(1, 2, 26, 19) == 1 * 100 + 1 * 10 + 3
    assert f_gen_age_exp(1, 2, 26, 24) == 1 * 100 + 1 * 10 + 4
    assert f_gen_age_exp(1, 2, 26, 29) == 1 * 100 + 1 * 10 + 5
    assert f_gen_age_exp(1, 2, 26, 50) == 1 * 100 + 1 * 10 + 6

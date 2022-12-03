import os
import pytest
import numpy as np
import pandas as pd
from scipy import stats as sps
from insolver.feature_monitoring import DiscreteHomogeneityTests
from insolver.model_tools import download_dataset


def gen_examples_discr(samp_size):
    # SIMPLE SYNTHETIC EXAMPLES
    examples = [
        (np.random.randint(0, 2, samp_size), np.random.randint(0, 2, samp_size), 'Same distributions'),
        (np.random.randint(0, 20, samp_size), np.random.randint(0, 20, samp_size), 'Same distributions'),
        (np.random.randint(0, 100, samp_size), np.random.randint(0, 100, samp_size), 'Same distributions'),
        (sps.poisson.rvs(0, 0.5, size=samp_size), sps.poisson.rvs(0, 0.5, size=samp_size), 'Same distributions'),
        (np.random.randint(0, 10, samp_size), np.random.randint(4, 12, samp_size), 'Different distributions'),
        (np.random.randint(0, 10, samp_size), np.random.randint(0, 50, samp_size), 'Different distributions'),
        (
            sps.poisson.rvs(0, 0.5, size=samp_size),
            sps.poisson.rvs(0, 1.0, size=samp_size),
            'Different distributions',
        ),
    ]

    # EXAMPLES FROM TEST FRAME
    # Download frame if necessary
    if not os.path.exists('tests/data/freMPL-R.csv'):
        download_dataset('freMPL-R', 'tests/data')
    df = pd.read_csv('tests/data/freMPL-R.csv', low_memory=False)

    # Simple check for categorical feat., no nans
    x = df['SocioCateg'].values.copy()
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with same percent of nans
    x = df['DeducType'].values.copy()
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with different percents of nans
    x = df['SocioCateg'].values.copy()
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    idx1 = np.random.choice(np.arange(len(x1)), len(x1) // 3, replace=False)
    idx2 = np.random.choice(np.arange(len(x2)), len(x2) // 10, replace=False)
    x1[idx1] = np.nan
    x2[idx2] = np.nan
    examples.append((x1, x2, 'Different distributions'))

    # EDA showed that married and lonely people prefer different car bodies
    x = df['VehBody'].values.copy()
    filt = ~df['VehBody'].isna()
    x1 = x[(df['MariStat'] == 'Other') & filt]
    x2 = x[(df['MariStat'] == 'Alone') & filt]
    examples.append((x1, x2, 'Different distributions'))

    # Same case with nans
    x = df['VehBody'].values.copy()
    x1 = x[df['MariStat'] == 'Other']
    x2 = x[df['MariStat'] == 'Alone']
    examples.append((x1, x2, 'Different distributions'))

    # Analogic case for driver age and social category
    x = df['SocioCateg'].values.copy()
    x1 = x[df['DrivAge'] >= 35]
    x2 = x[df['DrivAge'] < 35]
    examples.append((x1, x2, 'Different distributions'))

    # Same case with nans
    idx = np.random.choice(np.arange(len(x)), len(x) // 3, replace=False)
    x[idx] = np.nan
    x1 = x[df['DrivAge'] >= 35]
    x2 = x[df['DrivAge'] < 35]
    examples.append((x1, x2, 'Different distributions'))

    # EXAMPLES WITH SMALL AND SIGNIFICANT NOISES IN DISTRIBUTION
    # We expect our system to be tolerant to some small noises

    # Small noise
    values = np.arange(4)
    probs = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])

    noise, eps = np.zeros(4), 0.02
    noise[1:] = np.random.uniform(-1, 1, 3) * eps
    noise[0] = -noise.sum()

    rv = sps.rv_discrete(values=(values, probs))
    x1 = rv.rvs(size=samp_size)
    rv = sps.rv_discrete(values=(values, probs + noise))
    x2 = rv.rvs(size=samp_size)
    examples.append((x1, x2, 'Same distributions'))

    # Significant noise
    values = np.arange(5)
    probs1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs2 = np.array([0.15, 0.3, 0.1, 0.1, 0.35])

    rv = sps.rv_discrete(values=(values, probs1))
    x1 = rv.rvs(size=samp_size)
    rv = sps.rv_discrete(values=(values, probs2))
    x2 = rv.rvs(size=samp_size)
    examples.append((x1, x2, 'Different distributions'))

    # Delete test data
    os.remove('tests/data/freMPL-R.csv')
    return examples


@pytest.mark.parametrize('x1, x2, expected', gen_examples_discr(5000))
def test_discrete_tests_class(x1, x2, expected):
    homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
    test_res = homogen_tester.run_all(x1, x2)
    for res in test_res:
        assert res[-1] == expected


# Check psi with small difference (0.1 <= psi < 0.2)
def test_psi_discr_small_diff():
    values = np.arange(5)
    probs1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs2 = np.array([0.15, 0.28, 0.12, 0.15, 0.3])

    rv = sps.rv_discrete(values=(values, probs1))
    x1 = rv.rvs(size=5000)
    rv = sps.rv_discrete(values=(values, probs2))
    x2 = rv.rvs(size=5000)

    homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
    psi_res = homogen_tester.run_all(x1, x2)[-1]
    assert psi_res[-1] == 'Small difference'


# Check if class recognises too small data in input
def test_shape_error_discr():
    with pytest.raises(Exception):
        homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.array([]), np.array([]))

    with pytest.raises(Exception):
        homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.zeros([100]), np.ones([200]))

    with pytest.raises(Exception):
        homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.zeros([550]), np.ones([550]))


# Check if class recognises type missmatches
def test_type_error_discr():
    with pytest.raises(Exception):
        homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all([0] * 1000, [1] * 2000)

    with pytest.raises(Exception):
        homogen_tester = DiscreteHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.random.randint(0, 5, 1000), np.random.randn(2000))


# Check if class recognises bad hypeparameters
def test_attr_error_discr():
    with pytest.raises(Exception):
        _ = DiscreteHomogeneityTests(pval_thresh=1.02, samp_size=500, bootstrap_num=100)
    with pytest.raises(Exception):
        _ = DiscreteHomogeneityTests(pval_thresh=0.02, samp_size=90, bootstrap_num=100)

    with pytest.raises(Exception):
        _ = DiscreteHomogeneityTests(pval_thresh=0.02, samp_size=500, bootstrap_num=5)

import os
import pytest
import numpy as np
import pandas as pd
from scipy import stats as sps
from insolver.feature_monitoring import ContinuousHomogeneityTests
from insolver.feature_monitoring import psi_cont_2samp
from insolver.model_tools import download_dataset


def gen_examples_cont(samp_size):
    # SIMPLE SYNTHETIC EXAMPLES
    examples = [
        (np.random.uniform(0, 1, samp_size), np.random.uniform(0, 1, samp_size), 'Same distributions'),
        (np.random.randn(samp_size) + 1000, np.random.randn(samp_size) + 1000, 'Same distributions'),
        (sps.chi2.rvs(df=4, size=samp_size), sps.chi2.rvs(df=4, size=samp_size), 'Same distributions'),
        (sps.chi2.rvs(df=4, size=samp_size), sps.chi2.rvs(df=4, size=samp_size) * 5, 'Different distributions'),
        (sps.chi2.rvs(df=5, size=samp_size), sps.chi2.rvs(df=3, size=samp_size), 'Different distributions'),
        (np.random.randn(samp_size) * 10 + 1000, np.random.randn(samp_size) * 7 + 1000, 'Different distributions'),
        (np.random.uniform(0, 1, samp_size), np.random.uniform(0, 1.5, samp_size), 'Different distributions'),
    ]

    # EXAMPLES FROM TEST FRAME
    # Download frame if necessary
    if not os.path.exists('tests/data/freMPL-R.csv'):
        download_dataset('freMPL-R', 'tests/data')
    df = pd.read_csv('tests/data/freMPL-R.csv', low_memory=False)

    # Simple check for numerical feat., no nans
    x = df['DrivAge'].values.copy().astype(float)
    np.random.shuffle(x)
    x1 = x[: (len(x) // 2)]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with same percent of nans
    idx = np.random.choice(np.arange(len(x)), len(x) // 3, replace=False)
    x[idx] = np.nan
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with different percents of nans
    x = df['DrivAge'].values.copy().astype(float)
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    idx1 = np.random.choice(np.arange(len(x1)), len(x1) // 3, replace=False)
    idx2 = np.random.choice(np.arange(len(x2)), len(x2) // 10, replace=False)
    x1[idx1] = np.nan  # -1
    x2[idx2] = np.nan  # -1
    examples.append((x1, x2, 'Different distributions'))

    # Same for KBM (important case)
    x = df['BonusMalus'].values.copy().astype(float)
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with same percent of nans
    idx = np.random.choice(np.arange(len(x)), len(x) // 3, replace=False)
    x[idx] = np.nan
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Example with different percents of nans
    x = df['BonusMalus'].values.copy().astype(float)
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    idx1 = np.random.choice(np.arange(len(x1)), len(x1) // 3, replace=False)
    idx2 = np.random.choice(np.arange(len(x2)), len(x2) // 10, replace=False)
    x1[idx1] = np.nan
    x2[idx2] = np.nan
    examples.append((x1, x2, 'Different distributions'))

    # KBM for 2 groups of driver ages (expecting to get difference)
    x = df['BonusMalus'].values.copy().astype(float)
    x1 = x[df['DrivAge'] > 30]
    x2 = x[df['DrivAge'] <= 30]
    examples.append((x1, x2, 'Different distributions'))

    # Same case with nans
    idx = np.random.choice(np.arange(len(x)), len(x) // 3, replace=False)
    x1 = x[df['DrivAge'] > 30]
    x2 = x[df['DrivAge'] <= 30]
    examples.append((x1, x2, 'Different distributions'))

    # Case with low number of unique values
    x = df['RiskArea'].values.copy()
    np.random.shuffle(x)
    filt = ~df['RiskArea'].isna()
    x1 = x[filt][: len(x) // 2]
    x2 = x[filt][len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Same case with nans
    x = df['RiskArea'].values.copy()
    np.random.shuffle(x)
    x1 = x[: len(x) // 2]
    x2 = x[len(x) // 2 :]
    examples.append((x1, x2, 'Same distributions'))

    # Risk area for two types of social category (expecting to get difference)
    x = df['RiskArea'].values.copy()
    filt = ~df['RiskArea'].isna()
    x1 = x[filt & (df['SocioCateg'] == 'CSP1')]
    x2 = x[filt & (df['SocioCateg'] == 'CSP40')]
    examples.append((x1, x2, 'Different distributions'))

    # Same case with nans
    x = df['RiskArea'].values.copy()
    x1 = x[df['SocioCateg'] == 'CSP1']
    x2 = x[df['SocioCateg'] == 'CSP40']
    examples.append((x1, x2, 'Different distributions'))

    # Delete test data
    os.remove('tests/data/freMPL-R.csv')
    return examples


@pytest.mark.parametrize('x1, x2, expected', gen_examples_cont(5000))
def test_continuous_tests_class(x1, x2, expected):
    homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
    test_res = homogen_tester.run_all(x1, x2)
    for res in test_res:
        assert res[-1] == expected


# Check psi with small difference (0.1 <= psi < 0.2)
def test_psi_cont_small_diff():
    x1 = sps.chi2.rvs(df=4, size=5000)
    x2 = sps.chi2.rvs(df=5, size=5000)

    homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
    psi_res = homogen_tester.run_all(x1, x2)[-1]
    assert psi_res[-1] == 'Small difference'


# Check if class recognises too small data in input
def test_shape_error_cont():
    with pytest.raises(Exception):
        homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.array([]), np.array([]))

    with pytest.raises(Exception):
        homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.zeros([100]), np.ones([200]))

    with pytest.raises(Exception):
        homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.zeros([550]), np.ones([550]))


# Check if class recognises type missmatches
def test_type_error_cont():
    with pytest.raises(Exception):
        homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all([0] * 1000, [1] * 2000)

    with pytest.raises(Exception):
        homogen_tester = ContinuousHomogeneityTests(pval_thresh=0.05, samp_size=500, bootstrap_num=100)
        _ = homogen_tester.run_all(np.random.randint(0, 5, 1000), np.random.randn(2000))


# Check if class recognises bad hypeparameters
def test_attr_error_cont():
    with pytest.raises(Exception):
        _ = ContinuousHomogeneityTests(pval_thresh=1.02, samp_size=500, bootstrap_num=100)
    with pytest.raises(Exception):
        _ = ContinuousHomogeneityTests(pval_thresh=0.02, samp_size=90, bootstrap_num=100)

    with pytest.raises(Exception):
        _ = ContinuousHomogeneityTests(pval_thresh=0.02, samp_size=500, bootstrap_num=5)


# Check if nan_value for continuous psi test is always minimum in both arrays.
# It is important because we cannot fill nans with values
# which can seen in x1 or x2 as it will modify actual distributions.
def test_psi_cont_bad_nan():
    with pytest.raises(Exception):
        x1 = np.random.randn(1000)
        x2 = np.random.randn(1000)
        _ = psi_cont_2samp(x1, x2, nan_value=-1)

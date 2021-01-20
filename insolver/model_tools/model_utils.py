import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from numpy import log, sum
from sklearn.model_selection import train_test_split


def download_dataset(name, folder='datasets'):
    """Function for downloading and unzipping example datasets

    Args:
        name (str): Dataset name. Available datasets are freMPL-R and US_Accidents
        folder (str): Path to the folder to dataset saving

    Returns:
        str: Information about saved dataset

    """
    datasets = {
        'freMPL-R': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/freMPL-R.zip',
        'US_Accidents': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/US_Accidents_June20.zip',
    }
    if name not in datasets.keys():
        return f'Dataset {name} is not found. Available datasets are {", ".join(datasets.keys())}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    url = datasets[name]
    with urlopen(url) as file:
        with ZipFile(BytesIO(file.read())) as zfile:
            zfile.extractall(folder)

    return f'Dataset {name} saved to "{folder}" folder'


def train_val_test_split(*arrays, val_size, test_size, random_state=0, shuffle=True, stratify=None):
    """Function for splitting dataset into train/validation/test partitions.

    Args:
        *arrays (array_like): Arrays to split into train/validation/test sets containing predictors.
        val_size (float): The proportion of the dataset to include in validation partition.
        test_size (float): The proportion of the dataset to include in test partition.
        random_state (:obj:`int`, optional): Random state, passed to train_test_split() from scikit-learn. (default=0).
        shuffle (:obj:`bool`, optional): Passed to train_test_split() from scikit-learn. (default=True).
        stratify (:obj:`array_like`, optional): Passed to train_test_split() from scikit-learn. (default=None).

    Returns:
        tuple: (x_train, x_valid, x_test, y_train, y_valid, y_test).

        A tuple of partitions of the initial dataset.
    """
    n_arrays = len(arrays)
    split1 = train_test_split(*arrays, random_state=random_state, shuffle=shuffle,
                              test_size=test_size, stratify=stratify)
    if n_arrays > 1:
        train, test = split1[0::2], split1[1::2]
        split2 = train_test_split(*train, random_state=random_state, shuffle=shuffle,
                                  test_size=val_size / (1 - test_size), stratify=stratify)
        train, valid = split2[0::2], split2[1::2]
        return (*train, *valid, *test)
    else:
        train, test = split1[0], split1[1]
        split2 = train_test_split(train, random_state=random_state, shuffle=shuffle,
                                  test_size=val_size / (1 - test_size), stratify=stratify)
        train, valid = split2[0], split2[1]
        return train, valid, test


def train_test_column_split(x, y, df_column):
    """Function for splitting dataset into train/test partitions w.r.t. a column (pd.Series).

    Args:
        x (pd.DataFrame): DataFrame containing predictors.
        y (pd.DataFrame): DataFrame containing target variable.
        df_column (pd.Series): Series for train/test split, assuming it is contained in x.

    Returns:
        tuple: (x_train, x_test, y_train, y_test).

        A tuple of partitions of the initial dataset.
    """
    x1, y1, col_name = x.copy(), y.copy(), df_column.name
    y1[col_name] = df_column
    return (x1[x1[col_name] == 'train'].drop(col_name, axis=1), x1[x1[col_name] == 'test'].drop(col_name, axis=1),
            y1[y1[col_name] == 'train'].drop(col_name, axis=1), y1[y1[col_name] == 'test'].drop(col_name, axis=1))


def deviance_poisson(y_hat, y, weight=None):
    """Function for Poisson Deviance evaluation.

    Args:
        y_hat: Array with predictions.
        y: Array with target variable.
        weight: Weights for weighted metric.

    Returns:
        float, value of the Poisson deviance.
    """
    t_hat, t = y_hat + 1, y + 1
    if weight:
        return 2 * sum(weight*(t * log(t / t_hat) - (t - t_hat)))
    else:
        return 2 * sum(t * log(t / t_hat) - (t - t_hat))


def deviance_gamma(y_hat, y, weight=None):
    """Function for Gamma Deviance evaluation.

    Args:
        y_hat: Array with predictions.
        y: Array with target variable.
        weight: Weights for weighted metric.

    Returns:
        float, value of the Gamma deviance.
    """
    if weight:
        return 2 * sum(weight*(-log(y/y_hat) + (y-y_hat)/y_hat))
    else:
        return 2 * sum(-log(y/y_hat) + (y-y_hat)/y_hat)

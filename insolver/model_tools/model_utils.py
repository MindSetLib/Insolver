import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import xlogy
from pandas import DataFrame, Series, concat, qcut
from sklearn.model_selection import train_test_split


def download_dataset(name, folder='datasets'):
    """Function for downloading and unzipping example datasets

    Args:
        name (str): Dataset name. Available datasets are freMPL-R, US_Accidents and Lending_Club
        folder (str): Path to the folder to dataset saving

    Returns:
        str: Information about saved dataset

    """
    datasets = {
        'freMPL-R': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/freMPL-R.zip',
        'US_Accidents': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/US_Accidents_June20.zip',
        'US_Accidents_small': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.5/US_Accidents_small.zip',
        'Lending_Club': 'https://github.com/MindSetLib/Insolver/releases/download/v0.4.4/LendingClub.zip'
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
        if val_size != 0:
            split2 = train_test_split(*train, random_state=random_state, shuffle=shuffle,
                                      test_size=val_size / (1 - test_size), stratify=stratify)
            train, valid = split2[0::2], split2[1::2]
            return *train, *valid, *test
        else:
            return train, test
    else:
        train, test = split1[0], split1[1]
        if val_size != 0:
            split2 = train_test_split(train, random_state=random_state, shuffle=shuffle,
                                      test_size=val_size / (1 - test_size), stratify=stratify)
            train, valid = split2[0], split2[1]
            return train, valid, test
        else:
            return train, test


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


def deviance_poisson(y, y_pred, weight=None, agg='sum'):
    """Function for Poisson Deviance evaluation.

    Args:
        y: Array with target variable.
        y_pred: Array with predictions.
        weight: Weights for weighted metric.
        agg: Function to calculate deviance ['sum', 'mean'] or callable are supported.

    Returns:
        float, value of the Poisson deviance.
    """
    dict_func = {'sum': np.sum, 'mean': np.mean}
    func = dict_func[agg] if agg in ['sum', 'mean'] else agg if isinstance(agg, callable) else None
    if func is None:
        raise ValueError
    weight = 1 if weight is None else weight
    return func(2 * weight * (xlogy(y, y / y_pred) - (y - y_pred)))


def deviance_gamma(y, y_pred, weight=None, agg='sum'):
    """Function for Gamma Deviance evaluation.

    Args:
        y: Array with target variable.
        y_pred: Array with predictions.
        weight: Weights for weighted metric.
        agg: Function to calculate deviance ['sum', 'mean'] or callable are supported.

    Returns:
        float, value of the Gamma deviance.
    """
    dict_func = {'sum': np.sum, 'mean': np.mean}
    func = dict_func[agg] if agg in ['sum', 'mean'] else agg if isinstance(agg, callable) else None
    if func is None:
        raise ValueError
    weight = 1 if weight is None else weight
    return func(2 * weight * (np.log(y_pred/y) + y/y_pred - 1))


def inforamtion_value_woe(data, target, bins=10, cat_thresh=10, detail=False):
    """Function for Information value and Weight of Evidence computation.

    Args:
        data (pd.DataFrame): DataFrame with data to compute IV and WoE.
        target (:obj:`str` or :obj:`pd.Series`): Target variable to compute IV and WoE.
        bins (:obj:`int`, optional): Number of bins for WoE calculation for continuous variables.
        cat_thresh (:obj:`int`, optional): Maximum number of categories for non-binned WoE calculation.
        detail (:obj:`bool`, optional):  Whether to return detailed results DataFrame or not. Short by default.

    Returns:
        pd.DataFrame, DataFrame containing the data on Information Value (depends on detail argument).
    """
    detailed_result, short_result = DataFrame(), DataFrame()
    target = target.name if isinstance(target, Series) else target
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > cat_thresh):
            binned_x = qcut(data[ivars], bins,  duplicates='drop')
            d0 = DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp = DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        detailed_result = concat([detailed_result, temp], axis=0)
        short_result = concat([short_result, d], axis=0)
    return short_result if detail else detailed_result


def gain_curve(predict, exposure, step=1, figsize=(10, 6), gini_exact=False, output=False):
    gini_df = DataFrame()
    plt.figure(figsize=figsize)
    if isinstance(predict, (Series, np.ndarray)) and isinstance(exposure, Series):
        temp_df = concat([Series(predict, name='Predict').reset_index(drop=True),
                          exposure.reset_index(drop=True)], axis=1)
        temp_df = temp_df.sort_values('Predict', ascending=False).reset_index(drop=True)
        cumsum = temp_df.cumsum()
        normalized_df = cumsum/cumsum.loc[cumsum.index[-1]]
        if gini_exact:
            w = sum(temp_df[exposure.name])
            m = np.true_divide(sum(temp_df[exposure.name] * temp_df['Predict']), sum(temp_df[exposure.name]))
            temp_df['Rank'] = 0
            temp_df.loc[0, 'Rank'] = 1 + 0.5 * (temp_df.loc[0, exposure.name] - 1)
            for x in range(1, len(temp_df)):
                temp_df.loc[x, 'Rank'] = (temp_df.loc[x-1, 'Rank'] + 0.5 * (temp_df.loc[x-1, exposure.name] + 1)
                                          + 0.5 * (temp_df.loc[x, exposure.name] - 1))
            gini = 1 + 1/w - 2/(w**2 * m) * sum(temp_df[exposure.name] * temp_df['Predict'] * temp_df['Rank'])
        else:
            auc = (np.sum(normalized_df['Predict'] * np.append(np.diff(normalized_df[exposure.name]), 0))
                   + np.sum(np.append(np.diff(normalized_df['Predict']), 0) *
                            np.append(np.diff(normalized_df[exposure.name]), 0))/2)
            gini = 2 * auc - 1
        gini_df = gini_df.append(DataFrame.from_dict({'Gini': {'Predict': gini}}))
        plt.plot(normalized_df[exposure.name].values[::step], normalized_df['Predict'].values[::step],
                 label=f'Predict (Gini: {round(gini, 3)})')
    elif isinstance(predict, DataFrame) and isinstance(exposure, Series):
        temp_df = concat([predict.reset_index(drop=True), exposure.reset_index(drop=True)], axis=1)
        for pred_col in temp_df.columns[:-1]:
            temp_df2 = temp_df[[pred_col, exposure.name]].sort_values(pred_col, ascending=False).reset_index(drop=True)
            cumsum = temp_df2.cumsum()
            normalized_df = cumsum / cumsum.loc[cumsum.index[-1]]
            if gini_exact:
                w = sum(temp_df2[exposure.name])
                m = np.true_divide(sum(temp_df2[exposure.name] * temp_df2[pred_col]), sum(temp_df2[exposure.name]))
                temp_df2['Rank'] = 0
                temp_df2.loc[0, 'Rank'] = 1 + 0.5 * (temp_df2.loc[0, exposure.name] - 1)
                for x in range(1, len(temp_df2)):
                    temp_df2.loc[x, 'Rank'] = (temp_df2.loc[x-1, 'Rank'] + 0.5 * (temp_df2.loc[x-1, exposure.name] + 1)
                                               + 0.5 * (temp_df2.loc[x, exposure.name] - 1))
                gini = 1 + 1/w - 2/(w**2 * m) * sum(temp_df2[exposure.name] * temp_df2[pred_col] * temp_df2['Rank'])
            else:
                auc = (np.sum(normalized_df[pred_col] * np.append(np.diff(normalized_df[exposure.name]), 0))
                       + np.sum(np.append(np.diff(normalized_df[pred_col]), 0) *
                                np.append(np.diff(normalized_df[exposure.name]), 0))/2)
                gini = 2 * auc - 1
            gini_df = gini_df.append(DataFrame.from_dict({'Gini': {pred_col: gini}}))
            plt.plot(normalized_df[exposure.name].values[::step], normalized_df[pred_col].values[::step],
                     label=f'{pred_col} (Gini: {round(gini, 3)})')
    else:
        raise Exception
    plt.legend()
    plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), c='red', linestyle='--', linewidth=0.7)
    plt.title('Gains curve')
    plt.xlabel('Cumulative exposure')
    plt.ylabel('Cumulative response')
    plt.show()
    if output:
        return gini_df


def lift_score(predict, column, lift_type='groupby', q=10, output=False, reference='mean', kind='line', show=True):
    df = concat([column.reset_index(drop=True), Series(predict, name='Predict')], axis=1)
    if lift_type == 'groupby':
        pass
    elif lift_type == 'quantile':
        df[column.name] = qcut(column, q=q).reset_index(drop=True)
    else:
        raise Exception
    if reference == 'mean':
        df = df.groupby(column.name).mean() / np.mean(predict)
    elif reference == 'min':
        df = df.groupby(column.name).mean() / df.groupby(column.name).min()
    else:
        raise Exception
    if kind == 'bar':
        plt.bar(df.index.astype(str), height=df['Predict'])
    else:
        plt.plot(df.index.astype(str), df['Predict'])
    plt.title('Lift Metrics')
    plt.xlabel(column.name)
    plt.ylabel('Lift Score')
    plt.xticks(rotation=90)
    if show:
        plt.show()
    if output:
        return df

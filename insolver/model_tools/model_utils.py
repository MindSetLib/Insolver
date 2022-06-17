import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame, Series, concat, qcut, cut
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc


def train_val_test_split(*arrays, val_size, test_size, random_state=0, shuffle=True, stratify=None):
    """Function for splitting dataset into train/validation/test partitions.

    Args:
        *arrays (array_like): Arrays to split into train/validation/test sets containing predictors.
        val_size (float): The proportion of the dataset to include in validation partition.
        test_size (float): The proportion of the dataset to include in test partition.
        random_state (int, optional): Random state, passed to train_test_split() from scikit-learn. (default=0).
        shuffle (bool, optional): Passed to train_test_split() from scikit-learn. (default=True).
        stratify (array_like, optional): Passed to train_test_split() from scikit-learn. (default=None).

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
            return (*train, *valid, *test)
        else:
            return (*train, *test)
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


def inforamtion_value_woe(data, target, bins=10, cat_thresh=10, detail=False):
    """Function for Information value and Weight of Evidence computation.

    Args:
        data (pd.DataFrame): DataFrame with data to compute IV and WoE.
        target (str or pd.Series): Target variable to compute IV and WoE.
        bins (int, optional): Number of bins for WoE calculation for continuous variables.
        cat_thresh (int, optional): Maximum number of categories for non-binned WoE calculation.
        detail (bool, optional):  Whether to return detailed results DataFrame or not. Short by default.

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


def lorenz_curve(y_true, y_pred, exposure):
    """Calculating lorenz curve and Gini coefficient.

    Args:
        y_true: Array with target variable.
        y_pred: Array with predictions.
        exposure: Array with corresponding exposure

    Returns: cumulated_samples, cumulated_claim_amount, -minus_gini_coef

    """
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    exp = np.asarray(exposure)
    ranking = np.argsort(-pred)
    ranked_exposure, ranked_pure_premium = exp[ranking], true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    minus_gini_coef = 1 - 2 * auc(cumulated_samples, cumulated_claim_amount)
    return cumulated_samples, cumulated_claim_amount, -minus_gini_coef


def gain_curve(y_true, y_pred, exposure, step=1, figsize=(10, 6)):
    """Plot gains curve and calculate Gini coefficient. Mostly making use of
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html.

    Args:
        y_true: Array with target variable.
        y_pred: Array with predictions.
        exposure: Array with corresponding exposure
        step: Integer value which determines the increment between data indexes on which the gain curve will be
         evaluated.
        figsize: Tuple corresponding to matplotlib figsize.
    """

    plt.figure(figsize=figsize)
    plt.title('Gains curve')
    plt.xlabel('Fraction of policyholders\n (ordered by model from riskiest to safest)')
    plt.ylabel('Fraction of total claim amount')

    y_true = y_true[::step]

    # Random Baseline
    plt.plot([0, 1], [0, 1], c='red', linestyle='--', linewidth=0.5, label=f'Random Baseline')

    # Ideal Model
    cumul_samples, cumul_claim_amt, gini = lorenz_curve(y_true, y_true, exposure)
    plt.plot(cumul_samples, cumul_claim_amt, c='black', linestyle='-.', linewidth=0.5,
             label='Ideal Model (Gini: {:.3f})'.format(gini))

    # Fitted Models
    if isinstance(y_pred, list):
        names = [i for i in range(len(y_pred))]
        y_pred = [pred[::step] for pred in y_pred]
    elif isinstance(y_pred, DataFrame):
        names = y_pred.columns.tolist()
        y_pred = [y_pred[col].values[::step] for col in y_pred.columns]
    else:
        names = y_pred.name if (isinstance(y_pred, Series) and y_pred.name is not None) else '0'
        y_pred = y_pred[::step]
    if isinstance(y_pred, list):
        for i in range(len(y_pred)):
            cumul_samples, cumul_claim_amt, gini = lorenz_curve(y_true, y_pred[i], exposure)
            plt.plot(cumul_samples, cumul_claim_amt, label='Model {} (Gini: {:.3f})'.format(names[i], gini))
    else:
        cumul_samples, cumul_claim_amt, gini = lorenz_curve(y_true, y_pred, exposure)
        plt.plot(cumul_samples, cumul_claim_amt, label='Model {} (Gini: {:.3f})'.format(names, gini))
    plt.legend()
    plt.show()


def lift_score(predict, column, lift_type='groupby', q=10, output=False, reference='mean', kind='line', show=True):
    """

    Args:
        predict:
        column:
        lift_type:
        q:
        output:
        reference:
        kind:
        show:

    Returns:

    """
    df = concat([column.reset_index(drop=True), Series(predict, name='Predict').reset_index(drop=True)], axis=1)
    if lift_type == 'groupby':
        pass
    elif lift_type == 'quantile':
        df[column.name] = qcut(column, q=q).reset_index(drop=True)
    else:
        raise Exception
    if reference == 'mean':
        df = df.groupby(column.name).mean() / np.mean(predict)
    elif reference == 'min':
        df = df.groupby(column.name).mean() / df.groupby(column.name).mean().min()
    elif reference == 'max':
        df = df.groupby(column.name).mean() / df.groupby(column.name).mean().max()
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


def stability_index(scoring_variable, dev, oot, index='psi', binning_method='quantile', bins=10, detail=True):
    """Calculation of Population Stability Index or Characteristic Stability Index.
    Based on https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf,
    https://www.listendata.com/2015/05/population-stability-index.html and
    https://towardsdatascience.com/psi-and-csi-top-2-model-monitoring-metrics-924a2540bed8.

    Args:
        scoring_variable (str): The name of the variable with respect to which the index will be calculated.
        dev (pandas.DataFrame): The dataset containing `scoring_variable` on which the model was developed.
        oot (pandas.DataFrame): The out-of-time dataset containing `scoring_variable`.
        index (str): The type of stability index: Polulation (psi) or Characteristic (csi). Default 'psi'.
        binning_method (str): Method for splitting variable into bins, 'quantile' or 'equal_width'. Default 'quantile'.
         If scoring_variable is object or category column, then initial values are used, without any binning.
        bins (int): The number of bins the population will be divided into. Default 10.
        detail (bool): Whether to return detail info on index calculation or only index value.

    Returns:

    """
    assert index in ['psi', 'csi'], '"index" argument must be in ["psi", "csi"]'
    assert binning_method in ['quantile', 'equal_width'], ('"binning_method" argument must'
                                                           'be in ["quantile", "equal_width"]')
    assert ((scoring_variable in dev.columns) and
            (scoring_variable in oot.columns)), '"scoring_variable" must be in both `dev` and `out` datasets.'
    sc_var_dev, sc_var_oot = dev[scoring_variable], oot[scoring_variable]
    assert sc_var_dev.dtype == sc_var_oot.dtype, '"scoring_variable" type must be the same in both `dev` and `oot`'

    # if sc_var_dev.dtype in ['object', 'category']:
    #
    # else:
    #     if binning_method == 'quantile':
    #
    #     else:

    if index == 'psi':
        oot_bins = cut(sc_var_oot, bins=bins)
        dev_bins = cut(sc_var_dev, bins=oot_bins.cat.categories)
    else:
        dev_bins = cut(sc_var_dev, bins=bins)
        oot_bins = cut(sc_var_oot, bins=dev_bins.cat.categories)
    psi = concat([(oot_bins.value_counts().sort_index(ascending=False)/oot_bins.shape[0]*100).rename('OOT'),
                  (dev_bins.value_counts().sort_index(ascending=False)/dev_bins.shape[0]*100).rename('DEV')], axis=1)
    psi['Diff'] = psi['OOT'] - psi['DEV']
    psi['ln_OOT_DEV'] = np.log(psi['OOT']/psi['DEV'])
    psi['PSI'] = psi['Diff'] * psi['ln_OOT_DEV']
    total, total.loc[['ln_OOT_DEV', 'Diff']] = Series(np.sum(psi), name='Total'), '-'
    psi = psi.append(total)
    if detail:
        return psi
    else:
        return total['PSI']

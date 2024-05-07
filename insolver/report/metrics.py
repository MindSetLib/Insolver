from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
import numpy as np
import pandas as pd
from insolver.model_tools import (
    deviance_poisson,
    deviance_gamma,
    deviance_score,
    deviance_explained,
    deviance_explained_poisson,
    deviance_explained_gamma,
    lift_score,
)
from .error_handler import error_handler

gain_descr = 'gain curve description'
lift_descr = 'lift curve description'


@error_handler(True)
def _create_metrics_charts(X_train, X_test, y_train, y_test, predicted_train, predicted_test, exposure=None):
    descr_html = ''
    try:
        # calculate lift score
        train_lift = lift_score(predicted_train, y_train, show=False, output=True, lift_type='quantile', q=20)
        test_lift = lift_score(predicted_test, y_test, show=False, output=True, lift_type='quantile', q=20)
        # generate indexes to display %
        train_lift.index = np.arange(5, 105, 5)
        test_lift.index = np.arange(5, 105, 5)
        footer = {
            'train_lift': [list(train_lift.dropna().index), list(train_lift.dropna()['Predict'])],
            'test_lift': [list(test_lift.dropna().index), list(test_lift.dropna()['Predict'])],
            'y_name': y_train.name,
            'gain': 'false',
        }
        descr_html += f'''
        <div class="p-3 m-3 bg-light border rounded-3 fw-light">
            <h4 class="text-center fw-light">Lift Chart:</h4>
                <div id="lift_score"></div>
                <button class="btn btn-primary m-3" type="button" data-bs-toggle="collapse"
                data-bs-target="#collapse_lift" aria-expanded="False" aria-controls="collapseWidthExample">
                    Show description
                </button>
                <div class="collapse" id="collapse_lift">
                    <div class="p-3 m-3 bg-light border rounded-3 fw-light">
                    {lift_descr}</div>
                </div>
        </div>
        '''
    except ValueError:
        footer = {}
    gain = False
    # if exposure create gain curve
    if gain:
        footer['gain'] = 'true'

        # ideal model
        t1, t2, t3 = gini_coef(y_train, y_train, X_train[exposure])
        footer['train_ideal_gain'] = [list(t1), list(t2), t3]
        t1, t2, t3 = gini_coef(y_test, y_test, X_test[exposure])
        footer['test_ideal_gain'] = [list(t1), list(t2), t3]
        # real model
        t1, t2, t3 = gini_coef(y_train, predicted_train, X_train[exposure])
        footer['train_gain'] = [list(t1), list(t2), t3]
        t1, t2, t3 = gini_coef(y_test, predicted_test, X_test[exposure])
        footer['test_gain'] = [list(t1), list(t2), t3]

        descr_html += f'''
        <div class="p-3 m-3 bg-light border rounded-3 fw-light">
            <h4 class="text-center fw-light">Gain Curve:</h4>
                <div id="gini_score"></div>
                <button class="btn btn-primary m-3" type="button" data-bs-toggle="collapse"
                data-bs-target="#collapse_gain" aria-expanded="False" aria-controls="collapseWidthExample">
                    Show description
                </button>
                <div class="collapse" id="collapse_gain">
                    <div class="p-3 m-3 bg-light border rounded-3 fw-light">
                    {gain_descr}</div>
                </div>
        </div>'''

    return footer, descr_html


def _calc_psi(x_train, x_test, dataset):
    features = x_train.columns
    nav_items = ''
    tab_pane_items = ''

    for feature in features:
        # get x values from dataset using x_test and x_train indexes
        _x_train = dataset.loc[x_train.index]
        _x_test = dataset.loc[x_test.index]
        # if not categorical column, create psi
        if _x_train[feature].dtype != object:
            psi = stability_index(feature, _x_train, _x_test)
            nav_class = "nav-link active" if feature == features[0] else "nav-link"
            # replace ' ' so that href could work correctly
            feature_replaced = feature.replace(' ', '_')
            nav_items += f'''
            <li class="nav-item">
                <a class="{nav_class}" aria-current="true" href="#psi_{feature_replaced}" data-bs-toggle="tab">
                {feature}</a>
            </li>'''
            tab_pane_class = "tab-pane active" if feature == features[0] else "tab-pane"
            tab_pane_items += f'''
            <div class="{tab_pane_class}" id="psi_{feature_replaced}">
                {psi.to_html(classes = "table table-striped", justify="center")}
            </div>
            '''

    return f'''
    <div class="card text-center">
        <div class="card-header">
            <ul class="nav nav-tabs card-header-tabs text-nowrap p-3" data-bs-tabs="tabs"
            style="overflow-x: auto;">
                {nav_items}
            </ul>

        </div>
        <form class="card-body tab-content">
            {tab_pane_items}
        </form>
    </div>'''


def _calc_metrics(y_true, y_pred, task, metrics_to_calc, x, exposure=None):
    """Function to calculate metrics

    Args:
        y_true (1d array-like, or label indicator array / sparse matrix): Ground truth (correct) target values.
        y_pred (1d array-like, or label indicator array / sparse matrix): Estimated targets as returned by an
            estimator.
        metrics_to_calc: Names of metrics to calculate, can be 'all' (all metrics will be calculated), 'main' or list.

    Returns:
        dict: Where keys are metrics' names and values are scores.
    """
    result = dict()

    if task == "reg":
        functions = metrics_regression

        if metrics_to_calc == 'all':
            functions_names = metrics_regression.keys()
        elif metrics_to_calc == 'main':
            functions_names = functions_names_dict['reg_main']
        elif isinstance(metrics_to_calc, list):
            functions_names = metrics_to_calc
        else:
            raise TypeError(
                f'''{type(metrics_to_calc)} type of metrics_to_calc is not supported.
                            Must be "all", "main" or list.'''
            )

        result['root_mean_square_error'] = np.sqrt(functions['mean_squared_error'](y_true, y_pred))

    elif task == "class":
        # type_of_target can be continuous or binary
        type_of_true = type_of_target(y_true)
        type_of_pred = type_of_target(y_pred)
        functions = metrics_classification

        if type_of_true == 'binary' and type_of_pred == 'binary':
            if metrics_to_calc == 'all':
                functions_names = metrics_classification.keys()
            elif metrics_to_calc == 'main':
                functions_names = functions_names_dict['class_main']
            elif isinstance(metrics_to_calc, list):
                functions_names = metrics_to_calc
            else:
                raise TypeError(
                    f'''{type(metrics_to_calc)} type of metrics_to_calc is not supported.
                                Must be "all", "main" or list.'''
                )

        elif type_of_true == 'binary' and type_of_pred == 'continuous':
            functions_names = (
                functions_names_dict['binary_cont'] if not isinstance(metrics_to_calc, list) else metrics_to_calc
            )

        else:
            raise TypeError(f"Not supported target type <{type_of_true}> or predicted type <{type_of_pred}>")

    for name in functions_names:
        if name not in functions.keys():
            raise NotImplementedError(
                f'''{name} metric name is not supported. Supported names for {task} task:
                                      {functions.keys()}.'''
            )
        try:
            if name == 'gini_coef' and exposure:
                result[name] = functions[name](y_true, y_pred, x[exposure])[2]
            else:
                result[name] = functions[name](y_true, y_pred)

        except Exception as e:
            print(f'\t-{e}')

    return result


functions_names_dict = {
    'binary_cont': [
        "average_precision_score",
        "brier_score_loss",
        "det_curve",
        "hinge_loss",
        "log_loss",
        "precision_recall_curve",
        "roc_auc_score",
        "roc_curve",
    ],
    'class_main': [
        'roc_auc_score',
        'f1_score',
        'precision_score',
        'recall_score',
    ],
    'reg_main': [
        'mean_squared_error',
        'root_mean_squared_error',
        'gini_coef',
        'deviance_gaussian',
        'deviance_poisson',
        'deviance_gamma',
        'deviance_explained_gaussian',
        'deviance_explained_poisson',
        'deviance_explained_gamma',
        'mean_tweedie_deviance',
    ],
}


def gini_coef(true, pred, exp):
    true, pred = np.asarray(true), np.asarray(pred)
    exp = np.asarray(exp)
    ranking = np.argsort(-pred)
    ranked_exposure, ranked_pure_premium = exp[ranking], true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    gini = 2 * metrics.auc(cumulated_samples, cumulated_claim_amount) - 1
    return cumulated_samples, cumulated_claim_amount, gini


def stability_index(scoring_variable, dev, oot, kind='psi', bins=10):
    assert kind in ['psi', 'csi'], '"kind" argument must be in ["psi", "csi"]'
    if kind == 'psi':
        oot_bins = pd.cut(oot[scoring_variable], bins=bins)
        dev_bins = pd.cut(dev[scoring_variable], bins=oot_bins.cat.categories)
    else:
        dev_bins = pd.cut(dev[scoring_variable], bins=bins)
        oot_bins = pd.cut(oot[scoring_variable], bins=dev_bins.cat.categories)
    psi = pd.concat(
        [
            (oot_bins.value_counts().sort_index(ascending=False) / oot_bins.shape[0] * 100).rename('OOT'),
            (dev_bins.value_counts().sort_index(ascending=False) / dev_bins.shape[0] * 100).rename('DEV'),
        ],
        axis=1,
    )
    psi['Diff'] = psi['OOT'] - psi['DEV']
    psi['ln_OOT_DEV'] = np.log(psi['OOT'] / psi['DEV'])
    psi['ln_OOT_DEV'].replace([np.inf, -np.inf], 0, inplace=True)
    psi['PSI'] = psi['Diff'] * psi['ln_OOT_DEV']
    total, total.loc[['ln_OOT_DEV', 'Diff']] = pd.Series(np.sum(psi), name='Total'), '-'
    psi = psi.append(total).dropna()

    return psi


metrics_regression = {
    'explained_variance_score': metrics.explained_variance_score,
    'max_error': metrics.max_error,
    'mean_absolute_error': metrics.mean_absolute_error,
    'mean_squared_error': metrics.mean_squared_error,
    'root_mean_squared_error': lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
    'median_absolute_error': metrics.median_absolute_error,
    'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,
    'r2_score': metrics.r2_score,
    'deviance_gaussian': deviance_score,
    'deviance_poisson': deviance_poisson,
    'deviance_gamma': deviance_gamma,
    'deviance_explained_gaussian': deviance_explained,
    'deviance_explained_poisson': deviance_explained_poisson,
    'deviance_explained_gamma': deviance_explained_gamma,
    'mean_tweedie_deviance': metrics.mean_tweedie_deviance,
    'd2_tweedie_score': metrics.d2_tweedie_score,
    'mean_pinball_loss': metrics.mean_pinball_loss,
    'gini_coef': gini_coef,
    'lift_score': lift_score,
}

metrics_classification = {
    "accuracy_score": metrics.accuracy_score,
    "average_precision_score": metrics.average_precision_score,
    "balanced_accuracy_score": metrics.balanced_accuracy_score,
    "brier_score_loss": metrics.brier_score_loss,
    "classification_report": metrics.classification_report,
    "cohen_kappa_score": metrics.cohen_kappa_score,
    "confusion_matrix": metrics.confusion_matrix,
    "det_curve": metrics.det_curve,
    "f1_score": metrics.f1_score,
    "hamming_loss": metrics.hamming_loss,
    "hinge_loss": metrics.hinge_loss,
    "jaccard_score": metrics.jaccard_score,
    "log_loss": metrics.log_loss,
    "matthews_corrcoef": metrics.matthews_corrcoef,
    "multilabel_confusion_matrix": metrics.multilabel_confusion_matrix,
    "precision_recall_curve": metrics.precision_recall_curve,
    "precision_recall_fscore_support": metrics.precision_recall_fscore_support,
    "precision_score": metrics.precision_score,
    "recall_score": metrics.recall_score,
    "roc_auc_score": metrics.roc_auc_score,
    "roc_curve": metrics.roc_curve,
    "zero_one_loss": metrics.zero_one_loss,
}

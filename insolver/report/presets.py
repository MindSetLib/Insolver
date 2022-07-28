import pandas as pd
from insolver.wrappers import InsolverBaseWrapper, InsolverGBMWrapper, InsolverGLMWrapper
from sklearn.inspection import PartialDependenceDisplay
from .metrics import _calc_psi
import numpy as np
from shap import TreeExplainer, LinearExplainer
import lime.lime_tabular as lt
from .error_handler import error_handler


def _create_pandas_profiling():
    pandas_profiling = '''Generated profile report from a
        pandas <code>DataFrame</code> prepared by
        <code><a href="https://pypi.org/project/pandas-profiling/">Pandas profiling library</a></code>.
    '''
    return {
        'name': 'Pandas profiling',
        'parts': [
            '<div class="col-12"><button '
            'class="btn btn-primary" type="submit" '
            'onclick="window.location.href=\''
            './profiling_report.html\';">'
            'Go to report</button></div>'
        ],
        'header': f'<p class="fs-5 fw-light">{pandas_profiling}</p>',
        'footer': '',
        'icon': '<i class="bi bi-briefcase"></i>',
    }


@error_handler(False)
def _create_dataset_description(x_train, x_test, y_train, y_test, task, description, y_description, dataset=None):
    # calculate values for train/test split
    x_sum = len(x_train) + len(x_test)
    train_pct = round(len(x_train) / x_sum * 100)
    test_pct = round(len(x_test) / x_sum * 100)

    train_test_column = f'''
    <div class="col-3.ms-auto my-3">
        <div class="card">
            <div class="card-body ">
                <h4 class="card-title fw-light">Train / test split:</h4>
                <p class="card-text">in quantity: {len(x_train)} / {len(x_test)}</p>
                <p class="card-text">in %: {train_pct} % / {test_pct} %</p>
            </div>
        </div>
    </div>
    '''
    # create y description, contains specification, chart and values description
    described_y, footer = _describe_y(
        y_train,
        y_test,
        task,
        y_description,
        dataset,
        classes="table table-striped table-responsive-sm ",
        justify="center",
    )

    return {
        'name': 'Dataset description',
        'parts': [
            '<div class="p-3 my-3 bg-light border rounded-3 fs-5 fw-light">'
            f'{description}</div>'
            '<div class="p-3 my-3 bg-light border rounded-3 text-center fw-light">'
            '<div class="row row-cols-2 my-3 fs-6">'
            f'{train_test_column}'
            f'{described_y}'
            '</div>'
            '</div>'
        ],
        'header': '',
        'footer': footer,
        'icon': '<i class="bi bi-book"></i>',
    }


@error_handler(False)
def _create_importance_charts():
    # create html for js
    nav_items = ''
    tab_pane_items = ''
    for coef_name in ['relative_importance', 'scaled_importance', 'percentage']:
        nav_class = "nav-link active" if coef_name == 'relative_importance' else "nav-link"
        nav_items += f'''
        <li class="nav-item">
            <a class="{nav_class}" aria-current="true" href="#{coef_name}" data-bs-toggle="tab">
            {coef_name}</a>
        </li>'''
        tab_pane_class = "tab-pane active" if coef_name == 'relative_importance' else "tab-pane"
        tab_pane_items += f'''
        <div class="{tab_pane_class}" id="{coef_name}">
            <canvas id="chart_{coef_name}"></canvas>
        </div>
        '''
    return f'''
    <button class="btn btn-primary" type="button" data-bs-toggle="collapse" data-bs-target="#collapse_metrics"
     aria-expanded="False" aria-controls="collapseWidthExample">
        Show charts
    </button>
    <div class="collapse" id="collapse_metrics">
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
        </div>
    </div>
    '''


@error_handler(True)
def _create_shap(x_train, x_test, model, shap_type):
    # footer values are used by js in the report_template
    footer = dict()
    # save features names
    footer['features'] = list(x_train.columns)
    # check model type
    base_model = model.model if isinstance(model, InsolverBaseWrapper) else model
    linear_model = (
        model.model['glm'] if isinstance(model, InsolverGLMWrapper) and model.backend == 'sklearn' else base_model
    )
    for key, value in {'train': x_train, 'test': x_test}.items():
        # get shap values
        explainer = TreeExplainer(base_model) if shap_type == 'tree' else LinearExplainer(linear_model, value)
        shap_values = explainer.shap_values(value)
        shap_values = shap_values[0] if isinstance(shap_values, list) and (len(shap_values) == 2) else shap_values
        variables = list(value.columns)
        # get mean and create dict
        mean_shap = np.abs(shap_values).mean(axis=0).tolist()
        mean_dict = {variables[i]: mean_shap[i] for i in range(len(variables))}
        # sort dict
        mean_dict = dict(sorted(mean_dict.items(), key=lambda item: item[1]))
        # save mean dict items to create horizontal bar
        footer[f'mean_{key}'] = [list(mean_dict.keys()), [round(num, 3) for num in mean_dict.values()]]
        nav_items = ''
        tab_pane_items = ''
        # get 50 random values
        _value = value.sample(n=50) if len(value) > 50 else value
        _shap_values = explainer.shap_values(_value) if len(value) > 50 else shap_values
        # save shap for each feature
        for i in range(len(variables)):
            feat = variables[i]
            shap_feat = _shap_values.T[i]
            footer[f'{feat}_{key}'] = [
                [round(num, 4) for num in list(_value[feat])],
                [round(num, 4) for num in list(shap_feat)],
            ]
            # replace ' ' so that href could work correctly
            feature_replaced = feat.replace(' ', '_')
            nav_class = "nav-link active" if feat == footer['features'][0] else "nav-link"
            nav_items += f'''
            <li class="nav-item">
                <a class="{nav_class}" aria-current="true" href="#div_shap_{feature_replaced}" data-bs-toggle="tab">
                {feat}</a>
            </li>'''
            tab_pane_class = "tab-pane active" if feat == footer['features'][0] else "tab-pane"
            tab_pane_items += f'''
            <div class="{tab_pane_class}" id="div_shap_{feature_replaced}">
                <div id="shap_{feat}"></div>
            </div>
            '''

    return (
        footer,
        f'''
    <div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">
        <div id="shap_mean"></div>
    </div>
    <div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">
        <div class="card text-center">
            <div class="card-header">
                <ul class="nav nav-tabs card-header-tabs text-nowrap p-3" data-bs-tabs="tabs"
                 style="overflow-x: auto;">
                    {nav_items}
                </ul>

            </div>
            <form class="card-body tab-content d-flex justify-content-center">
                {tab_pane_items}
            </form>
        </div>
    </div>
    ''',
    )


@error_handler(True)
def _create_partial_dependence(x_train, x_test, model):
    # footer values are used by js in the report_template
    footer = {}
    model = model.model if isinstance(model, InsolverBaseWrapper) else model
    # get Partial Dependence
    pdp_train = PartialDependenceDisplay.from_estimator(
        estimator=model, X=x_train, features=x_train.columns, kind='average'
    ).pd_results
    pdp_test = PartialDependenceDisplay.from_estimator(
        estimator=model, X=x_test, features=x_test.columns, kind='average'
    ).pd_results
    # convert results to list
    for feat in pdp_train:
        feat['average'] = list(feat['average'][0])
        feat['values'] = list(feat['values'][0])
    for feat in pdp_test:
        feat['average'] = list(feat['average'][0])
        feat['values'] = list(feat['values'][0])

    # save features names
    footer['features'] = list(x_train.columns)
    footer['pdp_train'] = pdp_train
    footer['pdp_test'] = pdp_test

    nav_items = ''
    tab_pane_items = ''
    for feat in footer['features']:
        # replace ' ' so that href could work correctly
        feature_replaced = feat.replace(' ', '_')
        nav_class = "nav-link active" if feat == footer['features'][0] else "nav-link"
        nav_items += f'''
        <li class="nav-item">
            <a class="{nav_class}" aria-current="true" href="#div_pdp_{feature_replaced}" data-bs-toggle="tab">
            {feat}</a>
        </li>'''
        tab_pane_class = "tab-pane active" if feat == footer['features'][0] else "tab-pane"
        tab_pane_items += f'''
        <div class="{tab_pane_class}" id="div_pdp_{feature_replaced}">
            <div id="pdp_{feat}"></div>
        </div>
        '''
    return (
        footer,
        f'''
    <div class="card text-center">
        <div class="card-header">
            <ul class="nav nav-tabs card-header-tabs text-nowrap p-3" data-bs-tabs="tabs"
            style="overflow-x: auto;">
                {nav_items}
            </ul>
        </div>
        <form class="card-body tab-content d-flex justify-content-center">
            {tab_pane_items}
        </form>
    </div>''',
    )


@error_handler(False)
def _explain_instance(explain_instance, model, x, task, original_dataset, shap_type):
    footer = dict()
    footer['shap_waterfall'] = shap_explain(explain_instance, model, shap_type, x)
    footer['lime'] = lime_explain(explain_instance, x, model, task, original_dataset)
    return {
        'name': 'Explain instance',
        'parts': [
            '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
            '<h4 class="text-center fw-light">SHAP Waterfall chart:</h4>'
            '<div id="shap_waterfall"></div>'
            '</div>'
            '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
            '<h4 class="text-center fw-light">Lime chart:</h4>'
            '<div id="lime"></div>'
            '</div>'
        ],
        'header': '',
        'footer': footer,
        'icon': '<i class="bi bi-123"></i>',
    }


def _describe_y(y_train, y_test, task, y_description, dataset=None, **kwargs):
    descr_dict = {'Y specification:': y_description}
    # footer values are used by js in the report_template
    footer = {'task': task}
    # if dataset, create one y description
    if isinstance(dataset, pd.DataFrame):
        footer['type'] = 'dataset'
        try:
            y_column = dataset[y_train.name]
            if task == 'reg':
                descr = pd.DataFrame(round(y_column.describe(), 2))
                footer['data_y'] = list(y_column)
                descr_dict['Y chart:'] = '<div id="chart_y"></div>'

            else:
                descr = pd.DataFrame(y_column.value_counts())
                footer['index_y'] = list(pd.Series(list(descr.index)).apply(str))
                footer['data_y'] = list(descr[y_column.name])
                descr_dict['Y chart:'] = '<canvas id="chart_y"></canvas>'

            descr = descr.append(pd.Series({y_column.name: y_column.isnull().sum()}, name='null'))
            descr_dict['Y values description:'] = f'{descr.to_html(**kwargs)}'

        except KeyError:
            dataset = None

    # if not dataset, create train and test y descriptions
    elif dataset is None:
        footer['type'] = 'train_test'
        for key, value in {'train': y_train, 'test': y_test}.items():
            if task == 'reg':
                descr = pd.DataFrame(round(value.describe(), 2))
                footer[f'data_{key}'] = list(value)
                descr_dict['Y {key} chart:'] = '<div id="chart_y_{key}"></canvas>'

            else:
                descr = pd.DataFrame(value.value_counts())
                footer[f'index_{key}'] = list(pd.Series(list(descr.index)).apply(str))
                footer[f'data_{key}'] = list(descr[value.name])
                descr_dict[f'Y {key} chart:'] = f'<canvas id="chart_y_{key}"></canvas>'

            descr['null'] = value.isnull().sum()
            descr_dict['Y {key} values description:'] = '{descr.to_html(**kwargs)}'

    descr_html = ''

    for key in descr_dict:
        descr_html += f'''
            <div class="col-3.ms-auto my-3">
                <div class="card">
                    <div class="card-body p-3">
                        <h4 class="card-title fw-light">{key}</h4>
                        <p class="card-text">{descr_dict[key]}</p>
                    </div>
                </div>
            </div>
        '''

    return descr_html, footer


def _describe_dataset(x_train, x_test, dataset):
    # describe dataset for the 'Features values description'
    if isinstance(dataset, pd.DataFrame):
        description_table = (
            pd.concat([dataset.dtypes, dataset.isnull().sum(), round(dataset.describe().transpose(), 2)], axis=1)
            .rename(columns={0: 'type', 1: 'null'})
            .sort_values(by=['type'])
        )

        # Dataframe.describe() method is not working for object columns
        if description_table['count'].isnull().sum() > 0:
            description_table['count'] = dataset.count()
        description_table.fillna("-", inplace=True)

    # if dataset is none, description will be created for train and test values
    elif dataset is None:
        train_descr = pd.concat([round(x_train.describe().transpose(), 2), x_train.isnull().sum()], axis=1).rename(
            columns={0: 'null'}
        )
        test_descr = pd.concat([round(x_test.describe().transpose(), 2), x_test.isnull().sum()], axis=1).rename(
            columns={0: 'null'}
        )
        description_table = pd.concat([train_descr, test_descr], axis=1, keys=['Train', 'Test'])

    else:
        TypeError(f'Parameter `dataset` {type(dataset)} must be pandas.DataFrame.')

    return description_table


@error_handler(False)
def _create_features_description(x_train, x_test, dataset, description=None):
    # create html with features description
    html_grid = ''
    if description:
        if not (description, dict):
            raise NotImplementedError('Features description must be dict.')

        for key in description.keys():
            html_grid += f'''
            <div class="row">
                <div class="col-4 p-2 col-feat-wrapper text-dark text-center fw-bold">{key}</div>
                <div class="col-8 p-2 col-feat-wrapper text-dark fw-light">{description[key]}</div>
            </div>
            '''
    # get description table from _describe_dataset
    description_table = _describe_dataset(x_train, x_test, dataset)

    return {
        'name': 'Features description',
        'parts': [
            '<div class="p-3 my-3 bg-light border rounded-3 fw-light">'
            '<h4 class="text-center fw-light">Features specification:</h4>'
            f'{html_grid}'
            '</div>'
            '<div class="p-3 my-3 bg-light border rounded-3 text-center fw-light">'
            '<h4 class="text-center fw-light">Features values description:</h4>'
            f'{description_table.to_html(classes="table table-striped", justify="center")}'
            '</div>'
            '<div class="p-3 my-3 bg-light border rounded-3 text-center fw-light">'
            '<h4 class="text-center fw-light">Population Stability Index:</h4>'
            f'{_calc_psi(x_train, x_test, dataset)}'
            '</div>'
        ],
        'header': '',
        'footer': '',
        'icon': '<i class="bi bi-box-seam"></i>',
    }


def shap_explain(instance, model, shap_type, x):
    def logit(x):
        return np.true_divide(1, np.add(1, np.exp(x)))

    base_model = model.model if isinstance(model, InsolverBaseWrapper) else model
    linear_model = (
        model.model['glm'] if isinstance(model, InsolverGLMWrapper) and model.backend == 'sklearn' else base_model
    )
    explainer = TreeExplainer(base_model) if shap_type == 'tree' else LinearExplainer(linear_model, x)

    feature_names = list(instance.index)
    shap_values = explainer.shap_values(instance)
    cond_bool = isinstance(shap_values, list) and (len(shap_values) == 2)
    shap_values = shap_values[0] if cond_bool else shap_values
    expected_value = (
        explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    )

    prediction = pd.DataFrame(
        [expected_value] + shap_values.reshape(-1).tolist(), index=['E[f(x)]'] + feature_names, columns=['SHAP Value']
    )

    prediction['CumSum'] = np.cumsum(prediction['SHAP Value'])
    prediction['Value'] = np.append(np.nan, instance.values.reshape(-1))

    objective = base_model.objective if isinstance(base_model, InsolverGBMWrapper) else None
    link = None
    if objective is not None:
        link = np.exp if objective in ['poisson', 'gamma'] else logit if objective == 'binary' else None
    if link is not None:
        prediction['Link'] = link(prediction['CumSum'])
        prediction['Contribution'] = [link(expected_value[0])] + list(np.diff(prediction['Link']))
    else:
        prediction['Contribution'] = list(prediction['SHAP Value'])

    # return x, y and measure
    return [
        [round(i, 4) for i in list(prediction['Contribution'])],
        [
            prediction.index[i] if i == 0 else f'{prediction.index[i]}={instance.values.reshape(-1)[i - 1]:.3f}'
            for i in range(len(prediction.index))
        ],
        list(['relative'] * len(prediction)),
    ]


def lime_explain(instance, x, model, task, df):
    model = model.model if isinstance(model, InsolverBaseWrapper) else model
    # set lime parameters
    if task == 'reg':
        task = 'regression'
        predict_fn = model.predict
    else:
        task = 'classification'
        predict_fn = model.predict_proba
    categorical_features = []
    # find categorical columns
    for col in x.columns:
        if x[col].dtype in ['int8', 'int16', 'int32', 'int64'] and df[col].dtype == 'object':
            categorical_features.append(col)
    categorical_features = categorical_features if len(categorical_features) > 0 else None
    # create lime
    lime = lt.LimeTabularExplainer(
        training_data=x.to_numpy(), mode=task, feature_names=x.columns, categorical_features=categorical_features
    )
    # calculate explanation
    explanation = lime.explain_instance(data_row=instance, predict_fn=predict_fn)
    # as_list returns new names and scores
    list_expl = explanation.as_list()
    # as_map returns indexes and scores
    map_expl = explanation.as_map().values()
    first = next(iter(map_expl))
    result = [[], [], []]
    # save result
    for i in range(len(list_expl)):
        result[0].append(list_expl[i][0])
        result[1].append(round(list_expl[i][1], 3))
        col_index = first[i][0]
        result[2].append(f'Value = {round(instance[col_index], 3)}')
    # add predicted value
    result.append(round(explanation.predicted_value, 3))
    return result

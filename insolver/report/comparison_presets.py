from insolver.model_tools import ModelMetricsCompare
import numpy as np
import pandas as pd
import seaborn as sns
from .error_handler import error_handler


@error_handler(False)
def _create_models_comparison(
    x_train,
    y_train,
    x_test,
    y_test,
    dataset,
    task,
    models_to_compare,
    comparison_metrics,
    f_groups_type,
    f_bins,
    f_start,
    f_end,
    f_freq,
    p_groups_type,
    p_bins,
    p_start,
    p_end,
    p_freq,
    d_groups_type,
    d_bins,
    d_start,
    d_end,
    d_freq,
    model,
    main_diff_model,
    compare_diff_models,
    m_bins,
    m_freq,
    pairs_for_matrix,
    **kwargs,
):
    articles = []

    icons = {'train': '<i class="bi bi-clipboard"></i>', 'test': '<i class="bi bi-clipboard-check"></i>'}
    for key, value in {'train': [x_train, y_train], 'test': [x_test, y_test]}.items():
        # footer values are used by js in the report_template
        footer = {}
        # compare using ModelMetricsCompare and footer values
        footer[f'metrics_chart_{key}'], metrics = _get_ModelMetricsCompare(
            value[0], value[1], task, models_to_compare, comparison_metrics
        )
        # get features comparison and footer values
        footer[f'features_{key}'], feat_html_grid = _create_features_comparison(
            key, value[0], value[1], dataset, models_to_compare, f_groups_type, f_bins, f_start, f_end, f_freq
        )
        # get predict groups and footer values
        footer[f'predict_gp_{key}'], pr_gr_grid = _create_predict_groups(
            key, value[0], value[1], models_to_compare, p_groups_type, p_bins, p_start, p_end, p_freq
        )
        # get difference comparison and footer values
        footer[f'diff_{key}'], diff_grid = _create_difference_comparison(
            key,
            value[0],
            value[1],
            model,
            models_to_compare,
            main_diff_model,
            compare_diff_models,
            d_groups_type,
            d_bins,
            d_start,
            d_end,
            d_freq,
        )

        articles.append(
            {
                'name': f'Compare on {key} data',
                'parts': [
                    '<div class="p-3 m-3 bg-light border rounded-3 fw-light">'
                    '<h4 class="text-center fw-light">Metrics comparison chart:</h4>'
                    f'<canvas id="comparison_{key}"></canvas>'
                    '</div>'
                    '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
                    '<h4 class="text-center fw-light">Metrics comparison table:</h4>'
                    f'{metrics.to_html(**kwargs)}'
                    '</div>'
                    '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
                    '<h4 class="text-center fw-light">Predict groups chart:</h4>'
                    f'{pr_gr_grid}'
                    '</div>'
                    '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
                    '<h4 class="text-center fw-light">Difference chart:</h4>'
                    f'{diff_grid}'
                    '</div>'
                    '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
                    '<h4 class="text-center fw-light">Features comparison chart:</h4>'
                    f'{feat_html_grid}'
                    '</div>'
                    '<div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">'
                    '<h4 class="text-center fw-light">Comparison matrix:</h4>'
                    f'{_create_comparison_matrix(value[0], value[1], pairs_for_matrix, m_bins, m_freq)}'
                    '</div>'
                ],
                'header': '',
                'footer': footer,
                'icon': icons[key],
            }
        )

    return {
        'name': 'Compare models',
        'articles': articles,
        'icon': '<i class="bi bi-binoculars"></i>',
    }


def _get_ModelMetricsCompare(x, y, task, source, comparison_metrics):
    # use ModelMetricsCompare to create comparison dataframes
    mc = ModelMetricsCompare(x, y, task, source=source, metrics=comparison_metrics)
    mc.compare()
    metrics = mc.metrics_results
    metrics_columns = metrics.columns[2:]
    result = {'chart_name': 'Metrics comparison', 'models': list(metrics['Algo']), 'labels': list(metrics_columns)}
    for i in range(len(metrics)):
        row = metrics.iloc[i - 1]
        result[row['Algo']] = list(row[metrics_columns])

    return result, metrics


def _create_features_comparison(data_type, x, y, dataset, models_to_compare, groups_type, bins, start, end, freq):
    features = x.columns
    nav_items = ''
    tab_pane_items = ''
    result = {
        'features_names': list(features),
    }

    for feature in features:
        # get x values from dataset using x_test and x_train indexes
        _x = dataset.loc[x.index].drop([y.name], axis=1)
        x_y = pd.concat([_x, y], axis=1)
        models = []
        # create predict columns
        for model in models_to_compare:
            y_pred = model.predict(x)
            try:
                model_name = model.algo
            except AttributeError:
                model_name = model.__class__.__name__

            model_name = f'{model_name}_1' if model_name in x_y.columns else model_name
            x_y[model_name] = y_pred
            models.append(model_name)

        result['models'] = list(models)
        result['models'].append('target')

        x_y['group'] = _cut_column(x_y[feature], groups_type, bins, start, end, freq)

        # save the grouped feature and the count
        feature_groups_count = x_y[[feature, 'group']].groupby('group', as_index=False).count()
        result[f'{feature}_bins'] = list(feature_groups_count['group'].astype(str))
        result[f'{feature}_count'] = list(feature_groups_count[feature])

        # count models mean values in groups
        models_columns = list(models)
        models_columns.append('group')
        models_columns.append(y.name)
        feature_groups_mean = x_y[models_columns].groupby('group', as_index=False).mean()

        # save to result
        result[feature] = {'target': list(round(feature_groups_mean[y.name].fillna(0), 3))}
        for name in models:
            result[feature][name] = list(round(feature_groups_mean[name].fillna(0), 3))

        nav_class = "nav-link active" if feature == features[0] else "nav-link"
        # replace ' ' so that href could work correctly
        feature_replaced = feature.replace(' ', '_')
        nav_items += f'''
        <li class="nav-item">
            <a class="{nav_class}" aria-current="true" href="#comparison_{feature_replaced}_{data_type}"
            data-bs-toggle="tab">
            {feature}</a>
        </li>'''
        tab_pane_class = "tab-pane active" if feature == features[0] else "tab-pane"
        tab_pane_items += f'''
        <div class="{tab_pane_class}" id="comparison_{feature_replaced}_{data_type}">
            <div id="features_comparison_{feature}_{data_type}"></div>
        </div>
        '''

    return (
        result,
        f'''
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
    </div>''',
    )


def _create_predict_groups(data_type, x, y, models_to_compare, groups_type, bins, start, end, freq):
    nav_items = ''
    tab_pane_items = ''
    y_list = list(y)
    # save diag values as [y_list, y_list]
    result = {
        'diag': [y_list, y_list],
    }
    models = []
    for model in models_to_compare:
        try:
            model_name = model.algo
        except AttributeError:
            model_name = model.__class__.__name__
        y_pred = model.predict(x)
        # create Dataframe to save and group values
        df_y = pd.DataFrame(y.copy())
        model_name = f'{model_name}_1' if model_name in models else model_name
        df_y[model_name] = y_pred
        models.append(model_name)

        df_y['group'] = _cut_column(df_y[model_name], groups_type, bins, start, end, freq)

        # count predict results in groups
        model_groups_count = df_y[[model_name, 'group']].groupby('group', as_index=False).count()
        model_groups_count = model_groups_count[model_groups_count[model_name] != 0]

        result[f'{model_name}_count'] = list(model_groups_count[model_name])

        # mean predict and fact values
        model_groups_mean = df_y[[model_name, y.name, 'group']].groupby('group', as_index=False).mean()
        result[f'{model_name}_bins'] = list(model_groups_mean['group'].astype(str).dropna())
        result[f'{model_name}'] = [
            list(round(model_groups_mean[model_name].dropna(), 3)),
            list(round(model_groups_mean[y.name].dropna(), 3)),
        ]

        nav_class = "nav-link active" if model_name == models[0] else "nav-link"
        nav_items += f'''
        <li class="nav-item">
            <a class="{nav_class}" aria-current="true" href="#comparison_{model_name}_{data_type}" data-bs-toggle="tab">
            {model_name}</a>
        </li>'''
        tab_pane_class = "tab-pane active" if model_name == models[0] else "tab-pane"
        tab_pane_items += f'''
        <div class="{tab_pane_class}" id="comparison_{model_name}_{data_type}">
            <div id="models_comparison_{model_name}_{data_type}"></div>
        </div>
        '''
    result['models_names'] = models
    return (
        result,
        f'''
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
    </div>''',
    )


def _create_difference_comparison(
    data_type,
    x,
    y,
    main_model,
    models_to_compare,
    main_diff_model,
    compare_diff_models,
    groups_type,
    bins,
    start,
    end,
    freq,
):
    main_model = main_diff_model if main_diff_model else main_model
    models_to_compare = compare_diff_models if compare_diff_models else models_to_compare
    nav_items = ''
    tab_pane_items = ''
    result = {}
    models = []
    main_pred = main_model.predict(x)
    try:
        result['main_model'] = main_model.algo
    except AttributeError:
        result['main_model'] = main_model.__class__.__name__

    for model in models_to_compare:
        try:
            model_name = model.algo
        except AttributeError:
            model_name = model.__class__.__name__
        # create Dataframe to save and group values
        y_preds = pd.DataFrame(y.copy())
        y_preds['main_pred'] = main_pred
        y_pred = model.predict(x)
        model_name = f'{model_name}_1' if model_name in models else model_name
        y_preds[model_name] = y_pred
        models.append(model_name)

        # get the difference
        y_preds['diff_fact_model'] = y_preds[y.name] - y_preds['main_pred']
        y_preds['diff_model_model'] = y_preds[model_name] - y_preds['main_pred']

        y_preds['diff_groups'] = _cut_column(y_preds['diff_model_model'], groups_type, bins, start, end, freq)
        y_preds.sort_values(by='diff_model_model', inplace=True)

        # count in groups
        main_model_count = y_preds[['diff_model_model', 'diff_groups']].groupby('diff_groups', as_index=False).count()
        result[f'count_{model_name}'] = list(
            main_model_count[main_model_count['diff_model_model'] != 0]['diff_model_model']
        )

        # mean predict and fact values

        model_groups_mean = (
            y_preds[['diff_model_model', 'diff_fact_model', 'diff_groups']]
            .groupby('diff_groups', as_index=False)
            .mean()
        )

        result[f'diff_model_{model_name}'] = list(round(model_groups_mean['diff_model_model'].dropna(), 3))
        result[f'diff_fact_{model_name}'] = list(round(model_groups_mean['diff_fact_model'].dropna(), 3))
        result[f'{model_name}_bins'] = list(model_groups_mean['diff_groups'].astype(str).dropna())

        nav_class = "nav-link active" if model_name == models[0] else "nav-link"
        nav_items += f'''
        <li class="nav-item">
            <a class="{nav_class}" aria-current="true" href="#diff_comparison_{model_name}_{data_type}"
            data-bs-toggle="tab">
            {model_name}</a>
        </li>'''
        tab_pane_class = "tab-pane active" if model_name == models[0] else "tab-pane"
        tab_pane_items += f'''
        <div class="{tab_pane_class}" id="diff_comparison_{model_name}_{data_type}">
            <div id="models_diff_comparison_{model_name}_{data_type}"></div>
        </div>
        '''

    result['models_names'] = models
    return (
        result,
        f'''
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
    </div>''',
    )


def _create_comparison_matrix(x, y, pairs_for_matrix, bins, freq):
    if not pairs_for_matrix:
        return ''
    # check if pairs is [] and then make it [[]]
    pairs_for_matrix = [pairs_for_matrix] if len(np.array(pairs_for_matrix).shape) == 1 else pairs_for_matrix
    nav_items = ''
    tab_pane_items = ''
    cm = sns.light_palette("Blue", as_cmap=True)
    i = 0
    for pair in pairs_for_matrix:
        # check if pair is a pair
        if len(pair) != 2:
            raise NotImplementedError(
                f'Only two values is supported as a pair, now {pair} pair has {len(pair)} values.'
            )
        pair_df = pd.DataFrame(y)
        models_names = []
        for model in pair:
            try:
                model_name = model.algo
            except AttributeError:
                model_name = model.__class__.__name__
            # predict values
            pair_df[model_name] = model.predict(x)
            models_names.append(model_name)
        # create intervals with min and max values in all columns
        _start = min(pair_df.min()) - 1
        _end = max(pair_df.max()) + 1
        _bins = (
            pd.interval_range(start=_start, end=_end, freq=freq)
            if freq
            else pd.interval_range(start=_start, end=_end, periods=bins)
        )

        _bins = pd.IntervalIndex([pd.Interval(round(i.left, 2), round(i.right, 2), i.closed) for i in _bins])
        pair_df['groups'] = _cut_column(pair_df[y.name], groups_type='cut', bins=_bins)
        unique_gr = pair_df['groups'].unique()
        # create empty dataframes
        df_compare = pd.DataFrame(index=sorted(unique_gr), columns=sorted(unique_gr))
        df_count = pd.DataFrame(index=sorted(unique_gr), columns=sorted(unique_gr))
        # get rows where both predict values are in the interval
        for gr in unique_gr:
            for gr_2 in unique_gr:
                col0 = pair_df[models_names[0]]
                col1 = pair_df[models_names[1]]
                df_compare.loc[gr, gr_2] = (
                    pair_df.loc[col0.between(gr.left, gr.right) & col1.between(gr_2.left, gr_2.right), y.name].sum()
                    / pair_df.loc[col0.between(gr.left, gr.right) & col1.between(gr_2.left, gr_2.right), y.name].count()
                )
                df_count.loc[gr, gr_2] = pair_df.loc[
                    col0.between(gr.left, gr.right) & col1.between(gr_2.left, gr_2.right), y.name
                ].count()

        style_df = df_compare.style.background_gradient(axis=None, gmap=df_count, cmap=cm).format('{:.3f}')
        if len(pairs_for_matrix) == 1:
            return f'''<h5 class="text-center fw-light">{models_names[0]} and {models_names[1]} comparison matrix:</h5>
            {style_df.to_html(table_attributes='classes="table"')}'''
        else:
            nav_class = "nav-link active" if i == 0 else "nav-link"
            nav_items += f'''
            <li class="nav-item">
                <a class="{nav_class}" aria-current="true" href="#comparison_matrix_{i}" data-bs-toggle="tab">
                {models_names[0]} and {models_names[1]}</a>
            </li>'''
            tab_pane_class = "tab-pane active" if i == 0 else "tab-pane"

            tab_pane_items += f'''
            <div class="{tab_pane_class}" id="comparison_matrix_{i}">
            <h5 class="text-center fw-light">{models_names[0]} and {models_names[1]} comparison matrix:</h5>
                {style_df.to_html(table_attributes='classes="table"')}
            </div>
            '''
        i += 1
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


def _cut_column(column, groups_type, bins=None, start=None, end=None, freq=None):
    p = 0 if column.dtype == 'int64' else 2
    # group results
    try:
        groups_type, bins, start, end, freq = _get_columns_params(column, groups_type, bins, start, end, freq)
        if groups_type == 'cut':
            return pd.cut(column, bins)
        elif groups_type == 'qcut':
            return pd.qcut(column, bins)
        elif groups_type == 'freq':
            _start = min(column) - 1 if start is None else start
            _end = max(column) if end is None else end
            _bins = pd.interval_range(start=_start, end=_end, freq=freq)
            # count digits after the decimal point
            s = str(freq)
            p = 0 if s.isdecimal() else len(s.split('.')[1])
            _bins = pd.IntervalIndex([pd.Interval(round(i.left, p), round(i.right, p), i.closed) for i in _bins])
            # pd.interval_range creates big numbers, it rounds them
            return pd.cut(column, bins=_bins)

        else:
            raise NotImplementedError(
                f'`groups_type` = {groups_type} is not supported, must be `cut`, `qcut` or `freq`.'
            )
    except TypeError:
        return column


def _get_columns_params(column, groups_type, bins, start, end, freq):
    result = []
    for param in [groups_type, bins, start, end, freq]:
        if isinstance(param, dict):
            if column.name in param.keys():
                result.append(param[column.name])
            elif 'all' in param.keys():
                result.append(param['all'])
            else:
                raise NotImplementedError(f'Column name "{column.name}" or "all" keys must be implemented in {param}.')
        else:
            result.append(param)

    return tuple(result)

import glob
import pickle
import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from boosting_func import load_model  # scripts.

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
# import catboost as cgb

df, models_df, df_name = pd.DataFrame(), pd.DataFrame(), ''
models_dir = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Div([dcc.Upload(id='upload-df', children=html.Div(['Перетащите файл сюда, либо ',
                                                            html.A('выберите необходимый файл')]),
                         style={'margin': '1%', 'align': 'center', 'height': 'auto', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                'textAlign': 'center'})]),
    html.Div([dbc.Row([dbc.Col(html.Div('Dataset:')),
                       dbc.Col(html.Div('Model Dir:')),
                       dbc.Col(html.Div('Target:')),
                       dbc.Col(html.Div('External Prediction:')),
                       dbc.Col(html.Div('Column:')),
                       dbc.Col(html.Div('Split Column:')),
                       dbc.Col(html.Div('True Value Subset:')),
                       dbc.Col(html.Div('Exposure:'))]),
              dbc.Row([dbc.Col(html.Div(id='dataset_name')),
                       dbc.Col(dcc.Input(id='path_input')),
                       dbc.Col(dcc.Dropdown(id='drop_target')),
                       dbc.Col(dcc.Dropdown(id='drop_extern_pred')),
                       dbc.Col(dcc.Dropdown(id='drop_column')),
                       dbc.Col(dcc.Dropdown(id='drop_split')),
                       dcc.RadioItems(id='radio_list', value='all',
                                      options=[{'label': 'All', 'value': 'all'},
                                               {'label': 'Only Positive', 'value': 'pos'}]),
                       dbc.Col(dcc.Dropdown(id='drop_exposure'))])]),
    html.Div(id='output-graph')
])


def parse_contents(contents, filename):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        message = filename
        if filename.endswith('csv'):
            file = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('xls') or filename.endswith('xlsx'):
            file = pd.read_excel(io.BytesIO(decoded))
        else:
            file = None
            message = 'Выбран файл неверного расширения: ' + message
        globals()['df'] = file  # THIS IS NOT FINE!!!
        return [message]


@app.callback([Output('dataset_name', 'children'),
               Output('drop_column', 'options'),
               Output('drop_exposure', 'options'),
               Output('drop_extern_pred', 'options'),
               Output('drop_split', 'options')],
              [Input('upload-df', 'contents')],
              [State('upload-df', 'filename')])
def update_output(list_of_contents, list_of_names):
    name = parse_contents(list_of_contents, list_of_names) if list_of_contents is not None else df_name
    opt = [] if df.empty else [{'label': x, 'value': x} for x in df.columns]
    return [name, opt, opt, opt, opt]


@app.callback(Output('drop_target', 'options'),
              [Input('path_input', "value")])
def update_model_dir(value):
    if value is not None:
        m = [x.split('/')[-1].split('\\')[-1] for x in glob.glob(value + '/*.model')]
        m = list(set(['_'.join(x.split('_')[:-2]) for x in m]))
        m.sort()
        mdl = [{'label': x, 'value': x} for x in m]
        print('Making inference...')
        for target in m:
            models = [x for x in glob.glob(value + '/*.model') if target in x]
            for model in models:
                model_name = model.split('/')[-1].split('\\')[-1].split('.model')[0]
                try:
                    bst, params, target_name = load_model(model)
                except:
                    bst, params, target_name = load_sm(model)
                if type(bst) == xgb.Booster:
                    pred = bst.predict(xgb.DMatrix(df[[x for x in bst.feature_names if x in df.columns]]))
                elif type(bst) == lgb.Booster:
                    pred = bst.predict(df[[x for x in bst.feature_name() if x in df.columns]])
                else:
                    pred = np.exp(bst.predict(df[[x for x in bst.feature_names_ if x in df.columns]]))
                models_df[model_name] = pred
        print('Inference done.')
        return mdl
    else:
        return []


@app.callback(Output('output-graph', 'children'),
              [Input('drop_column', "value"),
               Input('drop_exposure', "value"),
               Input('path_input', "value"),
               Input('drop_target', "value"),
               Input('drop_extern_pred', "value"),
               Input('drop_split', "value"),
               Input('radio_list', "value")])
def update_graph(column, exposure, path, target, ext_pred, split, pos):
    train, test, g_train2, g_test2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    models_df_train, models_df_test = pd.DataFrame(), pd.DataFrame()
    fig2, fig3 = '', ''
    if (column is not None) and (exposure is not None) and (target is not None):
        models = [x for x in glob.glob(path + '/*.model') if target in x]
        # bst = unpickle(model, path)
        g_df = df[[column, exposure]].groupby(column).sum().reset_index()
        target_name = '_'.join(target.split('_')[1:])
        cols = [column, target_name, ext_pred] if ext_pred else [column, target_name]
        g_df3 = df[cols].groupby(column).mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=g_df[column], y=g_df[exposure], name=exposure))
        if split:
            pred_split = pd.concat([models_df, df[split, target_name]], axis=1)
            train, test = pred_split[pred_split[split] == 'train'], pred_split[pred_split[split] == 'test']
            models_df_train = models_df[models_df[split] == 'train']
            models_df_test = models_df[models_df[split] == 'test']
            g_train = train[[column, exposure]].groupby(column).sum().reset_index()
            g_test = test[[column, exposure]].groupby(column).sum().reset_index()
            g_train2 = train[cols].groupby(column).mean().reset_index()
            g_test2 = test[cols].groupby(column).mean().reset_index()
            fig2, fig3 = make_subplots(specs=[[{"secondary_y": True}]]), make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=g_train[column], y=g_train[exposure], name=exposure))
            fig3.add_trace(go.Bar(x=g_test[column], y=g_test[exposure], name=exposure))
        if pos == 'pos':
            g_df_pos = df.loc[df[target_name] > 0, [column, target_name]].groupby(column).mean().reset_index()
            fig.add_trace(go.Scatter(x=g_df_pos[column], y=g_df_pos[target_name], name='True'), secondary_y=True)
            if split:
                g_train3 = train.loc[train[target_name] > 0, [column, target_name]].groupby(column).mean().reset_index()
                g_test3 = test.loc[test[target_name] > 0, [column, target_name]].groupby(column).mean().reset_index()
                fig2.add_trace(go.Scatter(x=g_train3[column], y=g_train3[target_name], name='True'), secondary_y=True)
                fig3.add_trace(go.Scatter(x=g_test3[column], y=g_test3[target_name], name='True'), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(x=g_df3[column], y=g_df3[target_name], name='True'), secondary_y=True)
            if split:
                fig2.add_trace(go.Scatter(x=g_train2[column], y=g_train2[target_name], name='True'), secondary_y=True)
                fig3.add_trace(go.Scatter(x=g_test2[column], y=g_test2[target_name], name='True'), secondary_y=True)
        if ext_pred:
            fig.add_trace(go.Scatter(x=g_df3[column], y=g_df3[ext_pred], name='Prediction'), secondary_y=True)
            if split:
                fig2.add_trace(go.Scatter(x=g_train2[column], y=g_train2[ext_pred], name='Prediction'),
                               secondary_y=True)
                fig3.add_trace(go.Scatter(x=g_test2[column], y=g_test2[ext_pred], name='Prediction'), secondary_y=True)
        for model in models:
            model_name = model.split('/')[-1].split('\\')[-1].split('.model')[0]

            if pos == 'pos':
                g_df2 = pd.concat([df[[column, target_name]], models_df[model_name]], axis=1)
                g_df2[g_df2[target_name > 0]].groupby(column).mean().reset_index()
            else:
                g_df2 = pd.concat([df[column], models_df[model_name]], axis=1).groupby(column).mean().reset_index()
            fig.add_trace(go.Scatter(x=g_df2[column], y=g_df2[model_name], name='Prediction'), secondary_y=True)
            if split:
                if pos == 'pos':
                    g_train4 = pd.concat([train.loc[train[target_name] > 0, column],
                                          models_df_train.loc[models_df_train[target_name] > 0,
                                                              model_name]], axis=1).groupby(column).mean().reset_index()
                    g_test4 = pd.concat([test.loc[test[target_name] > 0, column],
                                         models_df_test.loc[models_df_test[target_name] > 0,
                                                            model_name]], axis=1).groupby(column).mean().reset_index()
                else:
                    g_train4 = pd.concat([train[column],
                                          models_df_train[model_name]], axis=1).groupby(column).mean().reset_index()
                    g_test4 = pd.concat([test[column],
                                         models_df_test[model_name]], axis=1).groupby(column).mean().reset_index()
                fig2.add_trace(go.Scatter(x=g_train4[column], y=g_train4[model_name],
                                          name='Prediction'), secondary_y=True)
                fig3.add_trace(go.Scatter(x=g_test4[column], y=g_test4[model_name],
                                          name='Prediction'), secondary_y=True)

        fig.update_layout(yaxis=dict(title_text='Sum of ' + exposure, side='right'),
                          yaxis2=dict(title_text='Mean Prediction', side='left'),
                          title='Overall Dataset')
        fig.update_xaxes(title_text=column)
        x = dcc.Graph(figure=fig)
        if split:
            fig2.update_layout(yaxis=dict(title_text='Sum of ' + exposure, side='right'),
                               yaxis2=dict(title_text='Mean Prediction', side='left'), title='Train Set')
            fig3.update_layout(yaxis=dict(title_text='Sum of ' + exposure, side='right'),
                               yaxis2=dict(title_text='Mean Prediction', side='left'), title='Test Set')
            fig2.update_xaxes(title_text=column)
            fig3.update_xaxes(title_text=column)
            x = [x, dcc.Graph(figure=fig2), dcc.Graph(figure=fig3)]
    else:
        x = []
    return x


def load_sm(model_name):
    with open(model_name.split('_sv.model')[0] + '_par.pickle', 'rb') as h:
        meta = pickle.load(h)
    model = xgb.Booster()
    model.load_model(model_name)
    model.feature_names = meta['feature_names']
    model.feature_types = meta['feature_types']
    return model, meta['hypepar'], meta['target']


if __name__ == '__main__':
    app.run_server(debug=True)

# def unpickle(model, path):
#     with open(path + '/' + model, 'rb') as h:
#         mdl = pickle.load(h)
#     return mdl

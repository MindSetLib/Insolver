import glob
import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.boosting_func import load_model

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

df, df_name = pd.DataFrame(), ''
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
                      dbc.Col(html.Div('Model:')),
                      dbc.Col(html.Div('Column:')),
                      dbc.Col(html.Div('Exposure:'))]),
              dbc.Row([dbc.Col(html.Div(id='dataset_name')),
                       dbc.Col(dcc.Input(id='path_input')),
                       dbc.Col(dcc.Dropdown(id='drop_model')),
                       dbc.Col(dcc.Dropdown(id='drop_column')),
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
            message = f'Выбран файл неверного расширения: {message}'
        globals()['df'] = file  # THIS IS NOT FINE!!!
        return [message]


@app.callback([Output('dataset_name', 'children'),
               Output('drop_column', 'options'),
               Output('drop_exposure', 'options')],
              [Input('upload-df', 'contents')],
              [State('upload-df', 'filename')])
def update_output(list_of_contents, list_of_names):
    name = parse_contents(list_of_contents, list_of_names) if list_of_contents is not None else df_name
    opt = [] if df.empty else [{'label': x, 'value': x} for x in df.columns]
    return [name, opt, opt]


@app.callback(Output('drop_model', 'options'),
              [Input('path_input', "value")])
def update_model_dir(value):
    if value is not None:
        m = [x.split('/')[-1].split('\\')[-1] for x in glob.glob(f'{value}/*.model')]
        return [{'label': x, 'value': x} for x in m]
    else:
        return []


@app.callback(Output('output-graph', 'children'),
              [Input('drop_column', "value"),
               Input('drop_exposure', "value"),
               Input('path_input', "value"),
               Input('drop_model', "value")])
def update_graph(column, exposure, path, model):
    if (column is not None) and (exposure is not None) and (model is not None):
        bst, params, target_name = load_model(f'{path}/{model}')
        if type(bst) == xgb.Booster:
            df['predict'] = bst.predict(xgb.DMatrix(df[[x for x in bst.feature_names if x in df.columns]]))
        elif type(bst) == lgb.Booster:
            df['predict'] = bst.predict(df[[x for x in bst.feature_name() if x in df.columns]])
        else:
            df['predict'] = np.exp(bst.predict(df[[x for x in bst.feature_names_ if x in df.columns]]))

        g_df = df[[column, exposure]].groupby(column).sum().reset_index()
        g_df2 = df[[column, 'predict']].groupby(column).mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=g_df[column], y=g_df[exposure], name=exposure))
        fig.add_trace(go.Scatter(x=g_df2[column], y=g_df2['predict'], name='Prediction'), secondary_y=True)
        fig.update_layout(yaxis=dict(title_text=f'Sum of {exposure}', side='right'),
                          yaxis2=dict(title_text='Mean Prediction', side='left'))
        fig.update_xaxes(title_text=column)
        x = dcc.Graph(figure=fig)
    else:
        x = []
    return x


if __name__ == '__main__':
    app.run_server(debug=True)

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

import pandas as pd
import xgboost as xgb
# import lightgbm as lgb
# import catboost as cgb

df = pd.DataFrame()
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
            message = 'Выбран файл неверного расширения: ' + message
        globals()['df'] = file  # THIS IS NOT FINE!!!
        return [message]


@app.callback([Output('dataset_name', 'children'),
               Output('drop_column', 'options'),
               Output('drop_exposure', 'options')],
              [Input('upload-df', 'contents')],
              [State('upload-df', 'filename')])
def update_output(list_of_contents, list_of_names):
    children = parse_contents(list_of_contents, list_of_names) if list_of_contents is not None else []
    opt = [] if df.empty else [{'label': x, 'value': x} for x in df.columns]
    return [children, opt, opt]


@app.callback(Output('drop_model', 'options'),
              [Input('path_input', "value")])
def update_model_dir(value):
    if value is not None:
        return [{'label': x.split('/')[-1].split('\\')[-1],
                 'value': x.split('/')[-1].split('\\')[-1]} for x in glob.glob(value + '/*.model')]
    else:
        return []


@app.callback(Output('output-graph', 'children'),
              [Input('drop_column', "value"),
               Input('drop_exposure', "value"),
               Input('path_input', "value"),
               Input('drop_model', "value")])
def update_table(column, exposure, path, model):
    if (column is not None) and (exposure is not None) and (model is not None):
        bst = unpickle(model, path)
        df['predict'] = bst.predict(xgb.DMatrix(df[bst.feature_names]))
        g_df = df[[column, exposure]].groupby(column).sum().reset_index()
        g_df2 = df[[column, 'predict']].groupby(column).mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=g_df[column], y=g_df[exposure], name=exposure))
        fig.add_trace(go.Scatter(x=g_df2[column], y=g_df2['predict'], name='Prediction'), secondary_y=True)
        fig.update_layout(yaxis=dict(title_text='Sum of ' + exposure, side='right'),
                          yaxis2=dict(title_text='Mean Prediction', side='left'))
        fig.update_xaxes(title_text=column)
        x = dcc.Graph(figure=fig)
    else:
        x = []
    return x


def unpickle(model, path):
    with open(path + '/' + model, 'rb') as h:
        mdl = pickle.load(h)
    return mdl


if __name__ == '__main__':
    app.run_server(debug=True)

import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

df = pd.DataFrame()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(id='upload-df',
               children=html.Div(['Перетащите файл сюда, либо ',
                                  html.A('выберите необходимый файл')]),
               style={'margin': '1%',
                      'align': 'center',
                      'height': 'auto',
                      'lineHeight': '60px',
                      'borderWidth': '1px',
                      'borderStyle': 'dashed',
                      'borderRadius': '5px',
                      'textAlign': 'center'}),
    html.Div(id='header'),
    dcc.Dropdown(id='drop_column'),
    html.Div(id='output-data'),
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
        return html.Div([html.H5(message)])


@app.callback([Output('header', 'children'),
               Output('drop_column', 'options')],
              [Input('upload-df', 'contents')],
              [State('upload-df', 'filename')])
def update_output(list_of_contents, list_of_names):
    children = parse_contents(list_of_contents, list_of_names) if list_of_contents is not None else []
    opt = [{'label': x, 'value': x} for x in df.columns] if not df.empty else []
    return [children, opt]


@app.callback(Output('output-data', 'children'),
              [Input('drop_column', "value")])
def update_table(value):
    if value is None:
        return []
    else:
        x = df[value].value_counts().sort_index().reset_index()
        return html.Div([dash_table.DataTable(id='table', columns=[{"name": i, "id": i} for i in x.columns],
                                              data=x.to_dict('records'))])


if __name__ == '__main__':
    app.run_server(debug=True)

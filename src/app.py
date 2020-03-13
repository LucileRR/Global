import datetime
import os
import yaml 

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



#Répertoire du fichier de données
PROCESSED_DIR = '../data/processed/'

# Table principale 
ALL_DATA_FILE =  'all_data.csv'

ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR, 
                         params['directories']['processed'], 
                         params['files']['all_data'])

# Lecture du fichiers des données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=["Last Update"])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df.day <= datetime.date(2020, 3,10)]
              )

countries = [{'label': c, 'value': c} for c in epidemie_df['Country/Region'].unique()]

app = dash.Dash('Covid-19 Explorer')
app.layout = html.Div([
    html.H1(['Covid-19 Explorer'], style={'textAlign':'center'}),
    html.Div([
        dcc.Dropdown(
            id='country',
            options=countries
        )
    ]),
    html.Div([
        dcc.Dropdown(
            id='country2',
            options=countries
        )
    ]),
    html.Div([
        dcc.RadioItems(
            id='variable',
            options=[
                {'label': 'Confirmed', 'value': 'Confirmed'},
                {'label': 'Deaths', 'value': 'Deaths'},
                {'label': 'Recovered', 'value': 'Recovered'}
            ],
            value='Confirmed'
        )
    ]),
    html.Div([
        dcc.Graph(id='graph1')
    ])
])

@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value')
    ]
)
def update_graph(country, country2, variable):
    print(country)
    
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )

    
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )
        ] if country2 is not None else [])
    }


if __name__ == '__main__':
    app.run_server(debug=True)
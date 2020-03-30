import datetime
import os
import yaml 

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
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
        ]),   
        dcc.Tab(label='Map', children=[
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        dcc.Tab(label='Model SIR', children=[
            html.Div([
                dcc.Dropdown(
                    id='country_model',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Input(
                    id='beta_input',
                    placeholder='Enter a value for beta...',
                    type='number'
                ),
                dcc.Input(
                    id='gamma_input',
                    placeholder='Enter a value for gamma...',
                    type='number'
                ),
                dcc.Input(
                    id='pop_input',
                    placeholder='Enter a value for population...',
                    type='number'
                ),
                dcc.Checklist(
                    id='opt_parameters',
                    options=[
                        {'label': 'Use optimal parameters', 'value': 'Y'}
                    ],
                    value=[]
                ),
            ]),
            html.Div([
                dcc.Graph(id='graph_model')
            ]),
        ]),
    ]),
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


@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day', 'value')
    ]
)
def update_map(map_day):  
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Longitude': 'mean','Latitude': 'mean', 'Confirmed': 'sum'})
              .reset_index()
             )

    return {
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed']/1_000, 10)   
                )
            )
        ],
        'layout': dict(
            geo=dict(showland=True)
        )
    }


# Fonction pour le modèle SIR
def SIR(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])


# Fonction d'optimisation des paramètres du modèle SIR
def sumsq_error(parameters):
    beta, gamma = parameters
    
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return([-beta*S*I, beta*S*I-gamma*I, gamma*I])

    solution = solve_ivp(SIR, [0, nb_steps-1], [pop_total, 1, 0], t_eval=np.arange(0, nb_steps, 1))
    
    return(sum((solution.y[1]-infected_population)**2))



@app.callback(
    Output('graph_model', 'figure'),
    [
        Input('country_model', 'value'),
        Input('beta_input', 'value'),
        Input('gamma_input', 'value'),
        Input('pop_input', 'value'),
        Input('opt_parameters', 'value')
    ]
)
def update_model(country_model, beta_input, gamma_input, pop_input, opt_parameters):
    global beta
    global gamma
    global nb_steps
    global pop_total
    global infected_population
    
    pop_total = pop_input
    
    if beta_input is None:
        beta = 0.01
    else:
        beta = beta_input
        
    if gamma_input is None:
        gamma = 0.1
    else:
        gamma = gamma_input
        
    if country_model is None:
        country_df = (epidemie_df
                      .groupby('day')
                      .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
                      .reset_index()
                     )
    else:
        country_df = (epidemie_df[epidemie_df['Country/Region'] == country_model]
                      .groupby(['Country/Region', 'day'])
                      .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
                      .reset_index()
                     )
        
    country_df['infected'] = country_df['Confirmed'].diff()
    nb_steps = country_df.shape[0]
    infected_population = country_df['infected'] 

    
    if 'Y' in opt_parameters and pop_total is not None:
        msol = minimize(sumsq_error, [beta, gamma], method='Nelder-Mead')
        print(msol.x)
        beta, gamma = msol.x
        solution = solve_ivp(SIR, [0, nb_steps-1], [0.00001*pop_total, 1, 0], t_eval=np.arange(0, nb_steps, 1))

    
    if pop_total is not None:
        solution = solve_ivp(SIR, [0, nb_steps-1], [0.00001*pop_total, 1, 0], t_eval=np.arange(0, nb_steps, 1))
    #plot_epidemia(solution, country_df.loc[2:]['infected'])
    
    return{
        'data': ([
            dict(
                x=solution.t,
                y=solution.y[0],
                type='line',
                name="Susceptible"
            )
        ] if pop_input is not None else []) +
        ([
            dict(
                x=solution.t,
                y=solution.y[1],
                type='line',
                name="Infected"
            )
        ] if pop_input is not None else []) +
        ([
            dict(
                x=solution.t,
                y=solution.y[2],
                type='line',
                name="Recovered"
            )
        ] if pop_input is not None else []) +
        [
            dict(
                x=country_df.loc[2:]['infected'].reset_index(drop=True).index,
                y=country_df.loc[2:]['infected'],
                type='line',
                name="Original Data"
            )
        ]
    }
    
if __name__ == '__main__':
    app.run_server(debug=True)
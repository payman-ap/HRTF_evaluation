# separate layout

# interactive_dash/layout.py

# Relative path definition
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


from dash import html, dcc
import dash_bootstrap_components as dbc


from scipy.io import loadmat
# Load the matrix from a .mat file
mat_data = loadmat(r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_hardware\SpeakerElevations_data.mat')
SpeakerElevations_data = mat_data['SpeakerElevations_data']


elevation_values = [int(val) for val in SpeakerElevations_data]


def get_layout():
    return dbc.Container([
        html.H2("Interactive HRTF Tool"),

        # Shared Controls (Path input and sliders)
        html.Div([
            html.Label("HRTF Folder Path"),
            dcc.Input(id="folder-path", type="text", placeholder="Paste path to .mat file folder", style={"width": "60%"}),
        ], style={'margin': '20px'}),

        html.Div([
            html.Label("Azimuth (°)"),
            dcc.Slider(
                id='azimuth-slider',
                min=0,
                max=360,
                step=5,
                value=0,
                marks={i: {'label': str(i), 'style': {'transform': 'rotate(90deg)', 'whiteSpace': 'nowrap'}} for i in range(0, 361, 45)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'margin': '20px'}),

        html.Div([
            html.Label("Elevation (°)"),
            dcc.Slider(
                id='elevation-slider',
                min=min(elevation_values),
                max=max(elevation_values),
                step=None,
                value=elevation_values[0],
                marks={val: {'label': str(val), 'style': {'transform': 'rotate(90deg)', 'whiteSpace': 'nowrap'}} for val in elevation_values},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'margin': '20px'}),

        # Tabs
        dbc.Tabs([
            dbc.Tab(label="HRTF Visualization", children=[
                html.Div([
                    html.Button("Plot", id='plot-btn', n_clicks=0),
                    html.Button("Add", id='add-btn', n_clicks=0, style={'marginLeft': '10px'}),
                ]),

                dcc.Checklist(id='trace-selector', value=[], inline=True),

                html.Div([
                    dcc.Graph(id='plot-left', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='plot-right', style={'display': 'inline-block', 'width': '49%'}),
                ])
            ]),

            dbc.Tab(label="Time Domain Visualization", children=[
                html.Div([
                    html.Button("Plot", id='plot-time-btn', n_clicks=0),
                    html.Button("Add", id='add-time-btn', n_clicks=0, style={'marginLeft': '10px'}),
                    dcc.Checklist(
                        id='time-options-checkbox',
                        options=[
                            {'label': 'Sync Y-axis Scales', 'value': 'sync_y'},
                            {'label': 'Log Y-axis', 'value': 'log_y'}
                        ],
                        value=[],
                        inline=True,
                        style={'marginLeft': '20px'}
                    ),
                ], style={'margin': '20px'}),

                dcc.Checklist(id='time-trace-selector', value=[], inline=True),

                html.Div([
                    dcc.Graph(id='time-plot-left', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='time-plot-right', style={'display': 'inline-block', 'width': '49%'}),
                ])
            ]),

            dbc.Tab(label="Elevational HRTFs", children=[
                html.Div([
                    html.Button("Plot Elevational HRTFs", id='plot-elevational-btn', n_clicks=0),
                    dcc.Checklist(
                        id='sync-colorscale-checkbox',
                        options=[{'label': 'Sync Color Scales', 'value': 'sync'}],
                        value=[],
                        inline=True,
                        style={'marginLeft': '20px'}
                    ),
                ], style={'margin': '20px'}),

                html.Div([
                    dcc.Graph(id='heatmap-left', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='heatmap-right', style={'display': 'inline-block', 'width': '49%'}),
                ])
            ]),

            dbc.Tab(label="Cues Visualization", children=[
                html.Div(id="audio-output", style={"marginTop": "20px", "color": "green"}),
                dcc.Upload(id="upload-audio", children=html.Button("Upload Audio")),
                html.Button("Play with HRTF", id="play-button"),
            ]),

            dbc.Tab(label="Compare To HRTF Estimations", children=[
                html.Div("Microphone input or live noise gen"),
                html.Button("Start Live Playback", id="live-playback-button"),
            ]),
        ])
    ])
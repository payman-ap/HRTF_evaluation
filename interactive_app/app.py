# Main Dash App
# interactive_dash/app.py
# Relative path definition
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# hrtf_eval_py/interactive_dash/app.py
import dash
from dash import html, dcc

import dash_bootstrap_components as dbc

import plotly.graph_objs as go
from hrtf_eval_py.interactive_app.layout import get_layout
from hrtf_eval_py.interactive_app.callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Interactive HRTF Plotter"
app.layout = get_layout()

register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)





# modularize callbacks

# Relative path definition
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# hrtf_eval_py/interactive_dash/callbacks.py
from dash import Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np
import os
from scipy.io import loadmat
from hrtf_eval_py.analysis.spectrum import real_fft_mag
from .state import trace_store

# Cache to hold data in memory
IR_data = None
fs = None
SourcePositions = None

# Time domain trace store
time_trace_store = []

def find_closest_index(azim, elev):
    diffs = (SourcePositions[:, 0] - azim) ** 2 + (SourcePositions[:, 1] - elev) ** 2
    return int(np.argmin(diffs))

def get_elevational_indices(azim):
    """Get all indices for a given azimuth across all elevations"""
    # Find all indices where azimuth matches (within tolerance)
    azim_tolerance = 2.5  # Tolerance for azimuth matching
    matching_indices = []
    matching_elevations = []
    
    for i in range(SourcePositions.shape[0]):
        az = SourcePositions[i, 0]  # First column: azimuth
        el = SourcePositions[i, 1]  # Second column: elevation
        if abs(az - azim) <= azim_tolerance:
            matching_indices.append(i)
            matching_elevations.append(el)
    
    # Sort by elevation
    sorted_pairs = sorted(zip(matching_elevations, matching_indices))
    elevations, indices = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(indices), list(elevations)

def register_callbacks(app):

    @app.callback(
        Output('azimuth-slider', 'min'),
        Output('azimuth-slider', 'max'),
        Output('elevation-slider', 'min'),
        Output('elevation-slider', 'max'),
        Input('folder-path', 'value'),
        prevent_initial_call=True
    )
    
    def load_hrtf_data(folder_path):
        global IR_data, fs, SourcePositions

        mat_file_path = os.path.join(folder_path, 'data_hrir_total.mat')
        data_loaded = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
        data = data_loaded['data_hrirs_total']
        IR_data = data.Data_IR
        fs = data.SamplingRate
        SourcePositions = data.SourcePositions

        return (int(SourcePositions[:, 0].min()), int(SourcePositions[:, 0].max()),
                int(SourcePositions[:, 1].min()), int(SourcePositions[:, 1].max()))

    @app.callback(
        Output('plot-left', 'figure'),
        Output('plot-right', 'figure'),
        Output('trace-selector', 'options'),
        Output('trace-selector', 'value'),
        Input('plot-btn', 'n_clicks'),
        Input('add-btn', 'n_clicks'),
        State('azimuth-slider', 'value'),
        State('elevation-slider', 'value'),
        State('trace-selector', 'value'),
        prevent_initial_call=True
    )
    def update_plot(n_plot, n_add, azim, elev, selected_traces):
        global IR_data, fs, SourcePositions
        triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

        if IR_data is None:
            return go.Figure(), go.Figure(), [], []

        index = find_closest_index(azim, elev)
        freq, mag_l = real_fft_mag(IR_data[index, 0, :], fs, db_scale=True)
        _, mag_r = real_fft_mag(IR_data[index, 1, :], fs, db_scale=True)
        label = f"{azim}°/{elev}°"

        if triggered == 'plot-btn':
            trace_store.clear()
            trace_store.append((label, freq, mag_l, mag_r))
        elif triggered == 'add-btn':
            trace_store.append((label, freq, mag_l, mag_r))

        fig_l = go.Figure()
        fig_r = go.Figure()
        selector_options = []
        selector_values = []

        for lbl, f, ml, mr in trace_store:
            selector_options.append({'label': lbl, 'value': lbl})
            if lbl in selected_traces or triggered != 'trace-selector':
                fig_l.add_trace(go.Scatter(x=f, y=ml, mode='lines', name=lbl))
                fig_r.add_trace(go.Scatter(x=f, y=mr, mode='lines', name=lbl))
                selector_values.append(lbl)

        fig_l.update_layout(title="Left Ear HRTF", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        fig_r.update_layout(title="Right Ear HRTF", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")

        return fig_l, fig_r, selector_options, selector_values

    @app.callback(
        Output('plot-left', 'figure', allow_duplicate=True),
        Output('plot-right', 'figure', allow_duplicate=True),
        Input('trace-selector', 'value'),
        prevent_initial_call=True
    )
    def toggle_traces(selected):
        fig_l = go.Figure()
        fig_r = go.Figure()

        for lbl, freq, mag_l, mag_r in trace_store:
            if lbl in selected:
                fig_l.add_trace(go.Scatter(x=freq, y=mag_l, mode='lines', name=lbl))
                fig_r.add_trace(go.Scatter(x=freq, y=mag_r, mode='lines', name=lbl))

        fig_l.update_layout(title="Left Ear HRTF", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        fig_r.update_layout(title="Right Ear HRTF", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")

        return fig_l, fig_r

    @app.callback(
        Output('heatmap-left', 'figure'),
        Output('heatmap-right', 'figure'),
        Input('plot-elevational-btn', 'n_clicks'),
        State('azimuth-slider', 'value'),
        State('sync-colorscale-checkbox', 'value'),
        prevent_initial_call=True
    )
    def update_elevational_heatmaps(n_clicks, azim, sync_colorscale):
        global IR_data, fs, SourcePositions
        
        if IR_data is None or n_clicks == 0:
            return go.Figure(), go.Figure()

        # Get all indices and elevations for the selected azimuth
        indices, elevations = get_elevational_indices(azim)
        
        if not indices:
            # No data found for this azimuth
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"No data found for azimuth {azim}°",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, empty_fig

        # Calculate magnitude responses for all elevations
        mag_data_left = []
        mag_data_right = []
        
        for idx in indices:
            freq, mag_l = real_fft_mag(IR_data[idx, 0, :], fs, db_scale=True)
            _, mag_r = real_fft_mag(IR_data[idx, 1, :], fs, db_scale=True)
            mag_data_left.append(mag_l)
            mag_data_right.append(mag_r)
        
        # Convert to numpy arrays for easier handling
        mag_data_left = np.array(mag_data_left)
        mag_data_right = np.array(mag_data_right)
        
        # Determine color scale limits
        if 'sync' in sync_colorscale:
            # Use the same scale for both ears
            vmin = min(np.min(mag_data_left), np.min(mag_data_right))
            vmax = max(np.max(mag_data_left), np.max(mag_data_right))
            colorscale_left = [vmin, vmax]
            colorscale_right = [vmin, vmax]
        else:
            # Use individual scales
            colorscale_left = [np.min(mag_data_left), np.max(mag_data_left)]
            colorscale_right = [np.min(mag_data_right), np.max(mag_data_right)]

        # Create heatmap for left ear
        fig_left = go.Figure(data=go.Heatmap(
            z=mag_data_left,
            x=freq,
            y=elevations,
            colorscale='viridis',
            zmin=colorscale_left[0],
            zmax=colorscale_left[1],
            colorbar=dict(title="Magnitude (dB)")
        ))
        
        fig_left.update_layout(
            title=f"Left Ear HRTF - Azimuth {azim}°",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Elevation (°)",
            xaxis_type="log" if freq[1] > freq[0] * 2 else "linear"  # Use log scale if appropriate
        )

        # Create heatmap for right ear
        fig_right = go.Figure(data=go.Heatmap(
            z=mag_data_right,
            x=freq,
            y=elevations,
            colorscale='viridis',
            zmin=colorscale_right[0],
            zmax=colorscale_right[1],
            colorbar=dict(title="Magnitude (dB)")
        ))
        
        fig_right.update_layout(
            title=f"Right Ear HRTF - Azimuth {azim}°",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Elevation (°)",
            xaxis_type="log" if freq[1] > freq[0] * 2 else "linear"  # Use log scale if appropriate
        )

        return fig_left, fig_right

    @app.callback(
        Output('time-plot-left', 'figure'),
        Output('time-plot-right', 'figure'),
        Output('time-trace-selector', 'options'),
        Output('time-trace-selector', 'value'),
        Input('plot-time-btn', 'n_clicks'),
        Input('add-time-btn', 'n_clicks'),
        State('azimuth-slider', 'value'),
        State('elevation-slider', 'value'),
        State('time-trace-selector', 'value'),
        State('time-options-checkbox', 'value'),
        prevent_initial_call=True
    )
    def update_time_plot(n_plot, n_add, azim, elev, selected_traces, time_options):
        global IR_data, fs, SourcePositions, time_trace_store
        triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

        if IR_data is None:
            return go.Figure(), go.Figure(), [], []

        index = find_closest_index(azim, elev)
        ir_left = IR_data[index, 0, :]
        ir_right = IR_data[index, 1, :]
        n_samples = len(ir_left)
        time_axis = np.arange(n_samples) / fs
        label = f"{azim}°/{elev}°"

        if triggered == 'plot-time-btn':
            time_trace_store.clear()
            time_trace_store.append((label, time_axis, ir_left, ir_right))
        elif triggered == 'add-time-btn':
            time_trace_store.append((label, time_axis, ir_left, ir_right))

        fig_l = go.Figure()
        fig_r = go.Figure()
        selector_options = []
        selector_values = []

        epsilon = 1e-12
        all_left_values = []
        all_right_values = []

        for lbl, t_axis, ir_l, ir_r in time_trace_store:
            selector_options.append({'label': lbl, 'value': lbl})
            if lbl in selected_traces or triggered != 'time-trace-selector':
                # Convert to dB
                ir_l_db = 20 * np.log10(np.maximum(np.abs(ir_l), epsilon))
                ir_r_db = 20 * np.log10(np.maximum(np.abs(ir_r), epsilon))
                fig_l.add_trace(go.Scatter(x=t_axis, y=ir_l_db, mode='lines', name=lbl))
                fig_r.add_trace(go.Scatter(x=t_axis, y=ir_r_db, mode='lines', name=lbl))
                selector_values.append(lbl)
                all_left_values.extend(ir_l_db)
                all_right_values.extend(ir_r_db)

        # Configure y-axis scaling
        y_axis_config = {}
        if 'sync_y' in time_options and all_left_values and all_right_values:
            y_min = min(min(all_left_values), min(all_right_values))
            y_max = max(max(all_left_values), max(all_right_values))
            y_axis_config['range'] = [y_min, y_max]

        fig_l.update_layout(
            title="Left Ear HRIR",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (dB)",
            yaxis=y_axis_config
        )
        fig_r.update_layout(
            title="Right Ear HRIR",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (dB)",
            yaxis=y_axis_config
        )

        return fig_l, fig_r, selector_options, selector_values


    @app.callback(
        Output('time-plot-left', 'figure', allow_duplicate=True),
        Output('time-plot-right', 'figure', allow_duplicate=True),
        Input('time-trace-selector', 'value'),
        Input('time-options-checkbox', 'value'),
        prevent_initial_call=True
    )
    def toggle_time_traces(selected, time_options):
        global time_trace_store

        fig_l = go.Figure()
        fig_r = go.Figure()

        epsilon = 1e-12
        all_left_values = []
        all_right_values = []

        for lbl, time_axis, ir_left, ir_right in time_trace_store:
            if lbl in selected:
                ir_l_db = 20 * np.log10(np.maximum(np.abs(ir_left), epsilon))
                ir_r_db = 20 * np.log10(np.maximum(np.abs(ir_right), epsilon))
                fig_l.add_trace(go.Scatter(x=time_axis, y=ir_l_db, mode='lines', name=lbl))
                fig_r.add_trace(go.Scatter(x=time_axis, y=ir_r_db, mode='lines', name=lbl))
                all_left_values.extend(ir_l_db)
                all_right_values.extend(ir_r_db)

        # Configure y-axis scaling
        y_axis_config = {}
        if 'sync_y' in time_options and all_left_values and all_right_values:
            y_min = min(min(all_left_values), min(all_right_values))
            y_max = max(max(all_left_values), max(all_right_values))
            y_axis_config['range'] = [y_min, y_max]

        fig_l.update_layout(
            title="Left Ear HRIR",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (dB)",
            yaxis=y_axis_config
        )
        fig_r.update_layout(
            title="Right Ear HRIR",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (dB)",
            yaxis=y_axis_config
        )

        return fig_l, fig_r

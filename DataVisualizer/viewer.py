import h5py
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import glob

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


def safe_log(arr):
    """
    Apply log transformation to array, replacing 0 with 1 before applying log
    :param arr: array to apply log transformation
    :return: log-transformed array
    """
    arr[arr == 0] = 1
    return np.log(arr)


# Function to map input value to slice index
def map_value_to_index(value):
    """
    Map value to slice index
    :param value: map value from -5 to 5
    :return: index of the slice
    """
    value = max(min(value, 5), -5)  # Limit value to range -5 to 5
    return int((value + 5) / 10 * 1023)


def run_viewer(file_path="data", debug=False):
    """
    Run the viewer
    :param file_path: Path to the data files (can be arbitrary large)
    :param debug: Chose debuging mode
    :return: None
    """
    files = glob.glob(file_path + "/*.h5")
    data_files = {}
    for file in files:
        data_files[file] = file

    # Initialize Dash app with a theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Layout of the app
    app.layout = html.Div([
        html.H1("Dataset Visualization", style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id='heatmap',
                style={'height': '80vh'},
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['selectbox']
                }
            ), width=8),
            dbc.Col([html.Div([
                html.Label('Select Dataset:'),
                dcc.Dropdown(
                    id='dataset-selector',
                    options=[{'label': k.split('\\')[-1], 'value': k} for k in data_files.keys()],
                    value=files[0]
                ),
            ], style={'marginBottom': '20'}),  # Add space below this block
                html.Div([
                    html.Label('Select Color Scheme:'),
                    dcc.Dropdown(
                        id='color-scheme-selector',
                        options=[
                            {'label': 'Blackbody', 'value': 'Blackbody'},
                            # {'label': 'Bluered', 'value': 'Bluered'},
                            # {'label': 'Blues', 'value': 'Blues'},
                            # {'label': 'Cividis', 'value': 'Cividis'},
                            # {'label': 'Earth', 'value': 'Earth'},
                            {'label': 'Electric', 'value': 'Electric'},
                            # {'label': 'Greens', 'value': 'Greens'},
                            # {'label': 'Greys', 'value': 'Greys'},
                            # {'label': 'Hot', 'value': 'Hot'},
                            {'label': 'Jet', 'value': 'Jet'},
                            # {'label': 'Picnic', 'value': 'Picnic'},
                            # {'label': 'Portland', 'value': 'Portland'},
                            {'label': 'Rainbow', 'value': 'Rainbow'},
                            # {'label': 'RdBu', 'value': 'RdBu'},
                            # {'label': 'Reds', 'value': 'Reds'},
                            {'label': 'Viridis', 'value': 'Viridis'},
                            # {'label': 'YlGnBu', 'value': 'YlGnBu'},
                            # {'label': 'YlOrRd', 'value': 'YlOrRd'}
                        ],
                        # Blackbody,Bluered,Blues,C ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd"""
                        value='Jet'
                    ),
                ], style={'marginBottom': '20'}),
                html.Div([
                    html.Label('Apply logarithmic transformation'),
                    html.Br(),  # Line break for better spacing
                    dcc.Checklist(
                        id='log-transform-check',
                        options=[
                            {'label': '', 'value': 'log'}
                        ],
                        value=[]
                    )
                ], style={'marginTop': '20px'}),
                html.Div([
                    html.Label('Set color-min:'),
                    dcc.Input(id='zmin-input', type='number', value=-2),
                ], style={'marginBottom': '20'}),  # Add space below this block
                html.Div([
                    html.Label('Set color-max:'),
                    dcc.Input(id='zmax-input', type='number', value=18),
                ], style={'marginBottom': '20'}),  # Add space below this block
                html.Div([
                    html.Label('Select plane:'),
                    dcc.RadioItems(
                        id='plane-selector',
                        options=[
                            {'label': 'h-l plane', 'value': 'XY'},
                            {'label': 'h-k plane', 'value': 'XZ'},
                            {'label': 'l-k plane', 'value': 'YZ'}
                        ],
                        value='XY',
                        inline=True
                    ),
                ], style={'marginBottom': '20'}),  # Add space below this block
                html.Div([
                    html.Label('Enter position (-5 to 5):'),
                    dcc.Input(
                        id='position-input',
                        type='number',
                        step=(10 / 1023),
                        value=0,
                        style={'marginRight': '10px'},
                    ),
                ], style={'marginTop': 20}),
                html.Div([
                    html.Label('Integration range:'),
                    dcc.Input(
                        id='select-integration',
                        type='number',
                        step=1,
                        value=5,
                        style={'marginRight': '10px'}
                    ),
                ], style={'marginTop': 20})], width=4),
        ]),
    ], style={'padding': '20px'})

    # Callback to update heatmap
    @app.callback(
        Output('heatmap', 'figure'),
        [Input('dataset-selector', 'value'),
         Input('position-input', 'value'),
         Input('plane-selector', 'value'),
         Input('color-scheme-selector', 'value'),
         Input('zmin-input', 'value'),
         Input('zmax-input', 'value'),
         Input('log-transform-check', 'value'),
         Input('select-integration', 'value')]
    )
    def update_figure(selected_dataset, input_value, selected_plane, color_scheme, zmin, zmax,
                      log_transform, integration_range):
        file = h5py.File(selected_dataset, 'r')
        dset = np.rot90(file['data'], k=1, axes=(0, 1))

        slice_index = map_value_to_index(input_value)

        if selected_plane == 'XY':
            slice_data = dset[:, :, slice_index - integration_range:slice_index + integration_range]
            slice_data = np.mean(slice_data, axis=2)
            x_name = "h (r.l.u)"
            y_name = "l (r.l.u)"
            slice_data = np.rot90(slice_data, k=1)
            plane_name = "h-l plane"
            # flip the x-axis
            slice_data = np.flip(slice_data, axis=0)
        elif selected_plane == 'XZ':
            slice_data = dset[:, slice_index - integration_range:slice_index + integration_range, :]
            slice_data = np.mean(slice_data, axis=1)
            x_name = "h (r.l.u)"
            y_name = "k (r.l.u)"
            slice_data = np.rot90(slice_data, k=1)
            plane_name = "h-k plane"
            # flip the x-axis
            slice_data = np.flip(slice_data, axis=0)
        else:
            slice_data = dset[slice_index - integration_range:slice_index + integration_range, :, :]
            slice_data = np.mean(slice_data, axis=0)
            x_name = "l (r.l.u)"
            y_name = "k (r.l.u)"
            slice_data = np.rot90(slice_data, k=1)
            # flip the x-axis
            slice_data = np.flip(slice_data, axis=0)
            plane_name = "l-k plane"
        # slice_data = np.nan_to_num(slice_data, nan=0.0)  # Replace NaN with 0
        # slice_data = np.rot90(slice_data, k=1)
        if 'log' in log_transform:
            slice_data = safe_log(slice_data)
        fig = go.Figure(
            go.Heatmap(z=slice_data,
                       x=np.linspace(-5, 5, 1023),
                       y=np.linspace(-5, 5, 1023),
                       colorscale=color_scheme,
                       zmin=zmin,
                       zmax=zmax
                       )
        )
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            yaxis_constrain='domain',
            xaxis_constrain='domain',
            xaxis=dict(
                title=x_name,
                ticktext=np.linspace(-5, 5, 1023),
                uirevision='constant'
            ),
            yaxis=dict(
                title=y_name,
                ticktext=np.linspace(-5, 5, 1023),
                uirevision='constant'
            ),
            title=f"{plane_name} Plane, Height: {input_value}",
        )
        fig.update_layout(clickmode='event+select')
        file.close()

        return fig
    if debug:
        app.run(debug=True)
    else:
        app.run(host="0.0.0.0", port="8050")


# Run the app
"""
if __name__ == '__main__':
    run_viewer()

"""
# 192.76.172.198
if __name__ == '__main__':
    run_viewer("E:/", False)



import base64
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import requests
import threading
import time

app = Dash(__name__)
server_url = "http://127.0.0.1:5000/get_intensity"
image_url = "http://127.0.0.1:5000/get_image"

intensity_value = {'intensity': 50}
fetching_enabled = {'enabled': True}

def fetch_intensity():

    while True:
        if fetching_enabled['enabled']:
            try:
                response = requests.get(server_url)
                if response.status_code == 200:
                    intensity_value['intensity'] = response.json()['intensity']
            except Exception as e:
                print(f"Error fetching intensity: {e}")
        time.sleep(1)


fetch_thread = threading.Thread(target=fetch_intensity, daemon=True)
fetch_thread.start()

app.layout = html.Div([
    html.H1("Welding Parameter Dashboard", style={
        'text-align': 'center',
        'margin-bottom': '20px',
        'font-family': 'Arial, sans-serif',
        'color': '#333',
        'background-color': '#f7f7f7',
        'padding': '10px',
        'border-radius': '8px',
        'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
    }),

    html.Div([
        html.Div([
            html.Div([
                html.Img(id='live-image', style={
                    'width': '150px',
                    'height': 'auto',
                    'margin-right': '15px',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                }),
                html.H3(id='intensity-display', style={
                    'margin': 'auto',
                    'text-align': 'center',
                    'font-size': '1.5em',
                    'color': '#555'
                }),
            ], style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-bottom': '15px',
                'background-color': '#ffffff',
                'padding': '15px',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
            }),

            dcc.Graph(id='3d-plot', config={'displayModeBar': False}, style={
                'height': 'calc(150% - 20px)',
                'width': '100%',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)',
                'margin': '0 auto',
                'position': 'relative',
                'top': '0',
                'transform': 'translateY(-0%)'
            }),
            dcc.Store(id='camera-store')
        ], style={
            'width': '68%',
            'height': '100%',
            'padding': '10px',
            'display': 'flex',
            'flex-direction': 'column',
            'justify-content': 'space-between'
        }),

        html.Div([
            dcc.Graph(id='2d-plot-current-voltage', style={
                'height': 'calc(33.33% - 10px)',
                'margin-bottom': '10px',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
            }),
            dcc.Graph(id='2d-plot-current-speed', style={
                'height': 'calc(33.33% - 10px)',
                'margin-bottom': '10px',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
            }),
            dcc.Graph(id='2d-plot-voltage-speed', style={
                'height': 'calc(33.33% - 10px)',
                'border-radius': '8px',
                'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
            }),
        ], style={
            'width': '30%',
            'padding': '10px',
            'border-left': '2px solid #ddd',
            'background-color': '#f9f9f9',
            'border-radius': '8px',
            'box-shadow': '0 2px 8px rgba(0, 0, 0, 0.1)',
            'height': '80vh',
            'display': 'flex',
            'flex-direction': 'column',
            'justify-content': 'space-between'
        }),
    ], style={
        'display': 'flex',
        'justify-content': 'space-between',
        'height': '80vh'
    }),

    dcc.Interval(
        id='interval-update',
        interval=1000,
        n_intervals=0
    )
], style={
    'font-family': 'Arial, sans-serif',
    'background-color': '#f0f2f5',
    'padding': '20px',
    'height': '100vh',
    'overflow': 'hidden'
})

@app.callback(
    [Output('3d-plot', 'figure'),
     Output('2d-plot-current-voltage', 'figure'),
     Output('2d-plot-current-speed', 'figure'),
     Output('2d-plot-voltage-speed', 'figure'),
     Output('camera-store', 'data')],
    [Input('interval-update', 'n_intervals')],
    [State('3d-plot', 'relayoutData'), State('camera-store', 'data')]
)
def update_plots(n, relayout_data, stored_camera):
    intensity = intensity_value['intensity']
    current_vals = np.linspace(40, 200, 50)
    voltage_vals = np.linspace(18, 23.8, 50)
    current_grid, voltage_grid = np.meshgrid(current_vals, voltage_vals)
    speed_grid = (current_grid * voltage_grid) / intensity

    # Maintain or update camera view
    camera = relayout_data.get('scene.camera') if relayout_data and 'scene.camera' in relayout_data else stored_camera

    fig3d = go.Figure(data=[go.Surface(
        x=current_grid, y=voltage_grid, z=speed_grid,
        colorscale='Viridis', opacity=0.6,
        showscale=False
    )])
    fig3d.update_layout(scene=dict(
        xaxis=dict(title='Current (A)', range=[40, 200]),
        yaxis=dict(title='Voltage (V)', range=[18, 23.8]),
        zaxis=dict(title='Speed (mm/s)', range=[20, 90]),
        camera=camera
    ))

    fig_cv = go.Figure(data=[go.Scatter(x=current_vals, y=intensity * 50 / current_vals)])
    fig_cv.update_layout(title="Current vs Voltage (Speed: 50 mm/s)", xaxis_title="Current (A)", yaxis_title="Voltage (V)")

    fig_cs = go.Figure(data=[go.Scatter(x=current_vals, y=(current_vals * 20) / intensity)])
    fig_cs.update_layout(title="Current vs Speed (Voltage: 20 V)", xaxis_title="Current (A)", yaxis_title="Speed (mm/s)")

    fig_vs = go.Figure(data=[go.Scatter(x=voltage_vals, y=(100 * voltage_vals) / intensity)])
    fig_vs.update_layout(title="Voltage vs Speed (Current: 100 A)", xaxis_title="Voltage (V)", yaxis_title="Speed (mm/s)")

    return fig3d, fig_cv, fig_cs, fig_vs, camera


@app.callback(
    Output('live-image', 'src'),
    Input('interval-update', 'n_intervals')
)
def update_image(n):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_image}"
    except Exception as e:
        print(f"Error fetching image: {e}")
    return None


@app.callback(
    Output('intensity-display', 'children'),
    Input('interval-update', 'n_intervals')
)
def update_intensity_display(n):
    return f"Current Intensity: {intensity_value['intensity']:.2f}"


@app.callback(
    Output('interval-update', 'disabled'),
    [Input('3d-plot', 'relayoutData')],
    [State('interval-update', 'disabled')]
)
def pause_fetching_during_interaction(relayout_data, is_disabled):
    if relayout_data and 'scene.camera' in relayout_data:
        return True
    return is_disabled

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

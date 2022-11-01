# import dash
from dash import Dash, Input, Output
from dash import html
from dash import dcc

from plotly import graph_objects as go

import numpy as np

from dashboard.utils import linear_layer, relu

app = Dash(__name__)

app.layout = html.Div([
    html.Div(html.H1("ReLU Visualization"),
             className="content_box",
             id="title_box"),
    html.Div([
        html.Div([
            html.Div([html.H1("W1"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="w1")],
                     className="parameter_box"),
            html.Div([html.H1("B1"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="b1")],
                     className="parameter_box"),
        ], className="layer_box", id="layer_1"),

        html.Div([
            html.Div([html.H1("W2"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="w2")],
                     className="parameter_box"),
            html.Div([html.H1("B2"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="b2")],
                     className="parameter_box"),
        ], className="layer_box", id="layer_2"),

        html.Div([
            html.Div([html.H1("W3"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="w3")],
                     className="parameter_box"),
            html.Div([html.H1("B3"),
                      dcc.Slider(min=-10.0, max=10.0,
                                 step=0.1, value=0.0,
                                 marks=None, tooltip={"placement": "bottom", "always_visible": True},
                                 updatemode="drag",
                                 className="parameter_slider", id="b3")],
                     className="parameter_box"),
        ], className="layer_box", id="layer_3")
    ], className="content_box", id="setings_box"),
    html.Div([dcc.Graph(id="layer_1_graph")],
             # dcc.Graph(id="layer_2_graph")],
             className="content_box", id="graph_box")
])


@app.callback(Output(component_id="layer_1_graph", component_property="figure"),
              [Input(component_id="w1", component_property="drag_value"),
               Input(component_id="b1", component_property="drag_value"),
               Input(component_id="w2", component_property="drag_value"),
               Input(component_id="b2", component_property="drag_value"),
               Input(component_id="w3", component_property="drag_value"),
               Input(component_id="b3", component_property="drag_value")
               ])
def create_figures(w1, b1, w2, b2, w3, b3):

    if w1 is None:
        w1 = 0.0

    if b1 is None:
        b1 = 0.0

    if w2 is None:
        w2 = 0.0

    if b2 is None:
        b2 = 0.0

    if w3 is None:
        w3 = 0.0

    if b3 is None:
        b3 = 0.0





    input_values = np.linspace(-5, 5, 1000)
    linear_output_1 = linear_layer(input_values, w=w1, b=b1)
    relu_output_1 = relu(linear_output_1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=input_values, y=relu_output_1,
                             mode="lines", name="ReLU output of layer 1"))

    linear_output_2 = linear_layer(relu_output_1, w=w2, b=b2)
    relu_output_2 = relu(linear_output_2)

    fig.add_trace(go.Scatter(x=input_values, y=relu_output_2,
                             mode="lines", name="ReLU output of layer 2"))

    linear_output_3 = linear_layer(relu_output_2, w=w3, b=b3)
    relu_output_3 = relu(linear_output_3)

    fig.add_trace(go.Scatter(x=input_values, y=relu_output_3,
                             mode="lines", name="ReLU output of layer 3"))

    fig.update_layout(yaxis_range=[-10, 10])

    return fig


if __name__ == "__main__":
    app.run(debug=True)

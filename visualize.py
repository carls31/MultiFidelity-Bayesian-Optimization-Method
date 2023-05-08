'''Author: Lorenzo Carlassara - mail: lorenzo.carlassara@mail.polimi.it
'''

import numpy as np
import plotly.graph_objs as go
from ipywidgets import interact, widgets

def update_layout_of_graph(fig: go.Figure,title: str = 'Plot')->go.Figure:
    fig.update_layout(
        width=800,
        height=600,
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title = 'input values',
                      yaxis_title = 'output values',
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      title={
                          'x': 0.5,
                          'xanchor': 'center'
                      })
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    return fig

def uncertainty_area_scatter(
        visible: bool = True,
        x_lines: np.array = np.array([]),
        y_upper: np.array = np.array([]),
        y_lower: np.array = np.array([]),
        name: str = "mean plus/minus standard deviation",
        legend_group: str = 'acqf',
        showlegend: bool = False,
) -> go.Scatter:

    return go.Scatter(
        visible=visible,
        x=np.concatenate((x_lines, x_lines[::-1])),  # x, then x reversed
        # upper, then lower reversed
        y=np.concatenate((y_upper, y_lower[::-1])),
        fill='toself',
        fillcolor='rgba(189,195,199,0.5)',
        line=dict(color='rgba(200,200,200,0)'),
        hoverinfo="skip",
        name= name,
        legendgroup = legend_group,
        showlegend=showlegend,
    )

def line_scatter(
    visible: bool = True,
    x_lines: np.array = np.array([]),
    y_lines: np.array = np.array([]),
    name_line: str = 'Predicted function',
    legend_group: str = 'acqf',
    color: str = 'blue',
    showlegend: bool = True,
) -> go.Scatter:
    # Adding the lines
    return go.Scatter(
        visible=visible,
        line=dict(color=color, width=2),
        x=x_lines,
        y=y_lines,
        name=name_line,
        legendgroup = legend_group,
        showlegend= showlegend
    )

def test_scatter(
    visible: bool = True,
    x_dots: np.array = np.array([]),
    y_dots: np.array = np.array([]),
    name_dots: str = 'Test',
    showlegend: bool = True
) -> go.Scatter:
    # Adding the dots
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        mode="markers",
        name=name_dots,
        marker=dict(color='green', size=7),
        showlegend=showlegend
    )

def dot_scatter(
    visible: bool = True,
    x_dots: np.array = np.array([]),
    y_dots: np.array = np.array([]),
    name_dots: str = 'Obs',
    legend_group: str = 'acqf',
    color: str = 'red',
    showlegend: bool = False
) -> go.Scatter:
    # Adding the dots
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        mode="markers",
        legendgroup = legend_group,
        name=name_dots,
        marker=dict(color=color, size=7),
        showlegend=showlegend
    )

import torch, gpytorch
def plot_GPR(data_x, data_y, model, x, legend_group, color=None, visible=True) -> list:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed = model(x)
        lower, upper = observed.confidence_region()
        data = []
        
        data.append(
            uncertainty_area_scatter(
                x_lines=x.numpy(),
                y_lower=lower.numpy(),
                y_upper=upper.numpy(),
                name=f"uncertainty",
                legend_group = legend_group,
                visible=visible))
        data.append(line_scatter(x_lines=x.numpy(), y_lines=observed.mean.numpy(), visible=visible, name_line =legend_group, color=color, legend_group = legend_group))
        data.append(dot_scatter(x_dots=data_x.numpy(), y_dots=data_y.numpy(), visible=visible, color=color, legend_group = legend_group))
        
    return data
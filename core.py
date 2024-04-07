# --------------------------------------------------------
# Calculations and plotting
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS as colors


def diversification(std, corr, N):
    ns = np.arange(1, N + 1)
    vs = ns * std * std
    cs = ns * (ns - 1) * corr * std * std
    xs = vs + cs
    return np.sqrt(xs) / (np.arange(1, N + 1))


def frontier_constants(means, covariance_matrix):
    cov_inv = np.linalg.inv(covariance_matrix)
    I = np.ones((means.shape[1], 1))
    A = np.vdot(np.dot(means, cov_inv), I)
    B = np.vdot(np.dot(means, cov_inv), means.T)
    C = np.vdot(np.dot(I.T, cov_inv), I)
    D = B * C - A * A
    return A, B, C, D


def frontier_return(A, B, C, D, std):
    variance = std**2
    return A / C + np.sqrt(variance * D / C - D / C**2)


def frontier_std(A, B, C, D, r):
    return np.sqrt(C / D * (r - A / C) ** 2 + 1 / C)


def slope_of_tangency_portfolio(means, covariance_matrix, rf):
    cov_inv = np.linalg.inv(covariance_matrix)
    excess_returns = means - rf
    return np.sqrt(np.vdot(np.dot(excess_returns, cov_inv), excess_returns.T))


def tangency_point(means, covariance_matrix, rf):
    cov_inv = np.linalg.inv(covariance_matrix)
    excess_returns = means - rf
    u = np.dot(excess_returns, cov_inv)
    w = u[0] / np.sum(u)
    mean = np.vdot(w, means)
    std = np.sqrt(np.vdot(np.dot(w, covariance_matrix), w))
    return mean, std


def plot_diversification(std, corr1, corr2, corr3, N):
    # Generate the data
    xs = diversification(std, corr1, N)
    ys = diversification(std, corr2, N)
    zs = diversification(std, corr3, N)

    # Labels for the lines based on the correlation values
    xs_label = f"ρ = {corr1:.2f}"
    ys_label = f"ρ = {corr2:.2f}"
    zs_label = f"ρ = {corr3:.2f}"

    # Create a figure
    fig = go.Figure()

    # Add the diversification lines for each correlation
    fig.add_trace(
        go.Scatter(x=np.arange(1, N + 1), y=zs * 100, mode="lines", name=zs_label)
    )
    fig.add_trace(
        go.Scatter(x=np.arange(1, N + 1), y=ys * 100, mode="lines", name=ys_label)
    )
    fig.add_trace(
        go.Scatter(x=np.arange(1, N + 1), y=xs * 100, mode="lines", name=xs_label)
    )

    # Set plot layout
    fig.update_layout(
        autosize=True,
        # title="Diversification Effect on Portfolio Standard Deviation",
        xaxis_title="Stocks",
        yaxis_title="σ",
        yaxis=dict(range=[0, 11]),  # Setting the y-axis range
        legend_title="Correlation",
        # width=800,
        # height=600,
    )
    return fig


def plot_mv(flags):
    stds = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    returns = np.array([0.075, 0.1, 0.16, 0.19, 0.25])
    traces = []
    num_traces = np.size(returns)

    if "2" in flags:
        pairs = np.array(
            [(i, j) for i, j in np.ndindex(num_traces, num_traces) if i < j]
        )
        for p in pairs:
            r1, r2 = returns[p[0]], returns[p[1]]
            means = np.array([[r1, r2]])
            cov = np.array([[stds[p[0]] ** 2, 0.0], [0.0, stds[p[1]] ** 2]])
            A, B, C, D = frontier_constants(means, cov)
            risky_returns = np.linspace(r1, r2, 100)
            frontier_stds = frontier_std(A, B, C, D, risky_returns)
            frontier = go.Scatter(
                x=frontier_stds * 100,
                y=risky_returns * 100,
                mode="lines",
                name="Frontier",
                marker=dict(color=colors[0]),
                line=dict(width=0.5),
            )
            traces.append(frontier)

    if "3" in flags:
        trips = np.array(
            [
                (i, j, k)
                for i, j, k in np.ndindex(num_traces, num_traces, num_traces)
                if i < j and j < k
            ]
        )
        for p in trips:
            r1, r2, r3 = returns[p[0]], returns[p[1]], returns[p[2]]
            means = np.array([[r1, r2, r3]])
            cov = np.array(
                [
                    [stds[p[0]] ** 2, 0.0, 0.0],
                    [0.0, stds[p[1]] ** 2, 0.0],
                    [0.0, 0.0, stds[p[2]] ** 2],
                ],
            )
            A, B, C, D = frontier_constants(means, cov)
            risky_returns = np.linspace(r1, r3, 100)
            frontier_stds = frontier_std(A, B, C, D, risky_returns)
            frontier = go.Scatter(
                x=frontier_stds * 100,
                y=risky_returns * 100,
                mode="lines",
                name="Frontier",
                marker=dict(color=colors[1]),
                line=dict(width=0.5),
            )
            traces.append(frontier)

    if "4" in flags:
        quads = np.array(
            [
                (i, j, k, l)
                for i, j, k, l in np.ndindex(
                    num_traces, num_traces, num_traces, num_traces
                )
                if i < j and j < k and k < l
            ]
        )
        for p in quads:
            r1, r2, r3, r4 = returns[p[0]], returns[p[1]], returns[p[2]], returns[p[3]]
            means = np.array([[r1, r2, r3, r4]])
            cov = np.array(
                [
                    [stds[p[0]] ** 2, 0.0, 0.0, 0.0],
                    [0.0, stds[p[1]] ** 2, 0.0, 0.0],
                    [0.0, 0.0, stds[p[2]] ** 2, 0.0],
                    [0.0, 0.0, 0.0, stds[p[3]] ** 2],
                ],
            )
            A, B, C, D = frontier_constants(means, cov)
            risky_returns = np.linspace(r1, r4, 100)
            frontier_stds = frontier_std(A, B, C, D, risky_returns)
            frontier = go.Scatter(
                x=frontier_stds * 100,
                y=risky_returns * 100,
                mode="lines",
                name="Frontier",
                marker=dict(color=colors[2]),
                line=dict(width=0.5),
            )
            traces.append(frontier)

    if "5" in flags:
        r1, r2, r3, r4, r5 = (
            returns[0],
            returns[1],
            returns[2],
            returns[3],
            returns[4],
        )
        means = np.array([[r1, r2, r3, r4, r5]])
        cov = np.array(
            [
                [stds[0] ** 2, 0.0, 0.0, 0.0, 0.0],
                [0.0, stds[1] ** 2, 0.0, 0.0, 0.0],
                [0.0, 0.0, stds[2] ** 2, 0.0, 0.0],
                [0.0, 0.0, 0.0, stds[3] ** 2, 0.0],
                [0.0, 0.0, 0.0, 0.0, stds[4] ** 2],
            ],
        )
        A, B, C, D = frontier_constants(means, cov)
        risky_returns = np.linspace(r1, r5, 100)
        frontier_stds = frontier_std(A, B, C, D, risky_returns)
        frontier = go.Scatter(
            x=frontier_stds * 100,
            y=risky_returns * 100,
            mode="lines",
            name="Frontier",
            marker=dict(color=colors[3]),
        )
        traces.append(frontier)

    layout = go.Layout(
        autosize=True,
        xaxis={"title": "σ - std dev"},
        yaxis={"title": "μ - return"},
        xaxis_range=[4, 31],
        yaxis_range=[4, 26],
        showlegend=False,
    )
    return go.Figure(data=traces, layout=layout)


def plot_frontier(means, cov, min_return, max_return, rf=None):
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(min_return, max_return, 100)
    stds = frontier_std(A, B, C, D, risky_returns)
    return_points = means[0]
    std_points = np.sqrt(np.diag(cov))

    # Create the main efficient frontier line
    frontier_trace = go.Scatter(
        x=stds * 100,
        y=risky_returns * 100,
        mode="lines",
        name="MVF",
        marker=dict(color=colors[0]),
    )

    # Initialize a list to collect all plot traces
    traces = [frontier_trace]

    # Add Capital Market Line if risk-free rate is provided
    if rf is not None:
        cml_returns = np.linspace(min_return, max_return, 100)
        slope = slope_of_tangency_portfolio(means, cov, rf)
        cml_stds = (cml_returns - rf) / slope
        cml_trace = go.Scatter(
            x=cml_stds * 100,
            y=cml_returns * 100,
            mode="lines",
            name="CML",
            marker=dict(color=colors[3]),
        )
        tangent = tangency_point(means, cov, rf)
        tp = go.Scatter(
            x=[tangent[1] * 100],
            y=[tangent[0] * 100],
            mode="markers",
            marker=dict(color=colors[3]),
            marker_size=8,
            name="TP",
        )
        traces.extend([cml_trace, tp])

    # Add points for risky assets
    if return_points.size > 1:
        risky_assets_trace = go.Scatter(
            x=std_points * 100,
            y=return_points * 100,
            mode="markers",
            marker=dict(color=colors[1]),
            marker_size=8,
            name="Stocks",
        )
        mvp_trace = go.Scatter(
            x=[np.sqrt(1 / C) * 100],
            y=[100 * A / C],
            mode="markers",
            marker=dict(color=colors[2]),
            marker_size=8,
            name="MVP",
        )
        traces.extend([risky_assets_trace, mvp_trace])

    # Define layout
    layout = go.Layout(
        autosize=True,
        xaxis={"title": "σ - std dev"},
        yaxis={"title": "μ - return"},
        xaxis_range=[0, 30],
        yaxis_range=[min_return * 100 - 1, max_return * 100],
        showlegend=True,
    )

    # Create figure with traces and layout
    return go.Figure(data=traces, layout=layout)

# --------------------------------------------------------
# Calculations and plotting
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
palette = sns.color_palette("colorblind")
sns.set_palette(palette)


def diversification(std, corr, N):
    ns = np.arange(1, N + 1)
    vs = ns * std * std
    cs = ns * (ns - 1) * corr * std * std
    xs = vs + cs
    return np.sqrt(xs) / (np.arange(1, N + 1))


def frontier_constants(means, covariance_matrix):
    cov_inv = np.linalg.inv(covariance_matrix)
    I = np.ones(len(means))
    A = means @ cov_inv @ I
    B = means @ cov_inv @ means
    C = I @ cov_inv @ I
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
    return np.sqrt(excess_returns @ cov_inv @ excess_returns)


def tangency_point(means, covariance_matrix, rf):
    cov_inv = np.linalg.inv(covariance_matrix)
    excess_returns = means - rf
    u = excess_returns @ cov_inv
    w = u / np.sum(u)
    mean = w @ means
    std = np.sqrt(w @ covariance_matrix @ w)
    return mean, std


def plot_diversification(std, corr1, corr2, corr3, N):
    xs = diversification(std, corr1, N)
    ys = diversification(std, corr2, N)
    zs = diversification(std, corr3, N)

    fig, ax = plt.subplots()

    ax.plot(np.arange(1, N + 1), zs * 100, label=f"{corr1:.2f}")
    ax.plot(np.arange(1, N + 1), ys * 100, label=f"{corr2:.2f}")
    ax.plot(np.arange(1, N + 1), xs * 100, label=f"{corr3:.2f}")

    ax.set_xlabel("Number of Stocks")
    ax.set_ylabel("σ")
    ax.set_ylim(0, 11)
    ax.legend(title="Correlations")
    return fig


def plot_frontier(means, cov, min_return, max_return, rf=None):
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(min_return, max_return, 100)
    stds = frontier_std(A, B, C, D, risky_returns)
    return_points = means
    std_points = np.sqrt(np.diag(cov))

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the mean-variance frontier
    ax.plot(stds * 100, risky_returns * 100, label="MVF")

    # Add points for risky assets and MVP
    if return_points.size > 1:
        ax.scatter(
            std_points * 100, return_points * 100, color="C1", label="Stocks", s=35
        )
        # Minimum Variance Portfolio point
        mvp_std = np.sqrt(1 / C) * 100
        mvp_return = 100 * A / C
        ax.scatter([mvp_std], [mvp_return], color="C2", label="MVP", s=35)

    # Add Capital Market Line if risk-free rate is provided
    if rf is not None:
        # Tangency point
        tangent = tangency_point(means, cov, rf)  # Assuming this function is defined

        ax.scatter([tangent[1] * 100], [tangent[0] * 100], color="C3", label="TP", s=35)
        cml_returns = np.linspace(min_return, max_return, 100)
        slope = slope_of_tangency_portfolio(
            means, cov, rf
        )  # Assuming this function is defined
        cml_stds = (cml_returns - rf) / slope
        ax.plot(cml_stds * 100, cml_returns * 100, color="C4", label="CML")

    # Set labels, title, and legend
    ax.set_xlabel("σ - std dev (%)")
    ax.set_ylabel("μ - return (%)")
    ax.set_xlim(0, 36)
    ax.set_ylim(min_return * 100 - 1, max_return * 100)
    ax.legend()

    return fig


def plot_mv(flags):
    stds = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    returns = np.array([0.075, 0.1, 0.16, 0.19, 0.25])
    num_traces = np.size(returns)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Pairs
    if "2" in flags:
        pairs = np.array(
            [(i, j) for i, j in np.ndindex(num_traces, num_traces) if i < j]
        )
        for p in pairs:
            plot_frontier_pair(ax, stds, returns, p)

    # Triples
    if "3" in flags:
        trips = np.array(
            [
                (i, j, k)
                for i, j, k in np.ndindex(num_traces, num_traces, num_traces)
                if i < j and j < k
            ]
        )
        for p in trips:
            plot_frontier_trip(ax, stds, returns, p)

    # Quadruples
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
            plot_frontier_quad(ax, stds, returns, p)

    # Quintuples
    plot_frontier_quint(ax, stds, returns)
    ax.scatter(stds * 100, returns * 100, s=25, color="C4")

    ax.set_xlabel("σ - std dev")
    ax.set_ylabel("μ - return")
    ax.legend()
    return fig


def plot_frontier_pair(ax, stds, returns, pair):
    r1, r2 = returns[pair[0]], returns[pair[1]]
    means = np.array([r1, r2])
    cov = np.diag([stds[pair[0]] ** 2, stds[pair[1]] ** 2])
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(r1, r2, 100)
    frontier_stds = frontier_std(A, B, C, D, risky_returns)
    ax.plot(frontier_stds * 100, risky_returns * 100, linewidth=0.5, color="C0")


def plot_frontier_trip(ax, stds, returns, trip):
    r1, r2, r3 = returns[trip[0]], returns[trip[1]], returns[trip[2]]
    means = np.array([r1, r2, r3])
    cov = np.diag([stds[trip[0]] ** 2, stds[trip[1]] ** 2, stds[trip[2]] ** 2])
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(r1, r3, 100)
    frontier_stds = frontier_std(A, B, C, D, risky_returns)
    ax.plot(frontier_stds * 100, risky_returns * 100, linewidth=0.5, color="C1")


def plot_frontier_quad(ax, stds, returns, quad):
    r1, r2, r3, r4 = (
        returns[quad[0]],
        returns[quad[1]],
        returns[quad[2]],
        returns[quad[3]],
    )
    means = np.array([r1, r2, r3, r4])
    cov = np.diag(
        [stds[quad[0]] ** 2, stds[quad[1]] ** 2, stds[quad[2]] ** 2, stds[quad[3]] ** 2]
    )
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(r1, r4, 100)
    frontier_stds = frontier_std(A, B, C, D, risky_returns)
    ax.plot(frontier_stds * 100, risky_returns * 100, linewidth=0.5, color="C2")


def plot_frontier_quint(ax, stds, returns):
    r1, r2, r3, r4, r5 = (
        returns[0],
        returns[1],
        returns[2],
        returns[3],
        returns[4],
    )
    means = np.array([r1, r2, r3, r4, r5])
    cov = np.diag(
        [stds[0] ** 2, stds[1] ** 2, stds[2] ** 2, stds[3] ** 2, stds[4] ** 2]
    )
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(r1, r5, 100)
    frontier_stds = frontier_std(A, B, C, D, risky_returns)
    ax.plot(
        frontier_stds * 100, risky_returns * 100, color="black", label="All 5 Stocks"
    )


def crra(x, gamma):
    if gamma == 1:
        return np.log(x)
    return (x ** (1 - gamma) - 1) / (1 - gamma)


def crra_inv(u, gamma):
    if gamma == 1:
        return np.exp(u)
    eta = 1 - gamma
    return (eta * u + 1) ** (1 / eta)


def expected_utility(uf, po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5):
    return (
        prob1 * uf(po1)
        + prob2 * uf(po2)
        + prob3 * uf(po3)
        + prob4 * uf(po4)
        + prob5 * uf(po5)
    )


def expected_value(po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5):
    return prob1 * po1 + prob2 * po2 + prob3 * po3 + prob4 * po4 + prob5 * po5


def std(po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5):
    m = expected_value(po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5)
    return np.sqrt(
        prob1 * (po1 - m) ** 2
        + prob2 * (po2 - m) ** 2
        + prob3 * (po3 - m) ** 2
        + prob4 * (po4 - m) ** 2
        + prob5 * (po5 - m) ** 2
    )


def moment(k, po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5):
    m = expected_value(po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5)
    s = std(po1, po2, po3, po4, po5, prob1, prob2, prob3, prob4, prob5)
    return (
        prob1 * ((po1 - m) / s) ** k
        + prob2 * ((po2 - m) / s) ** k
        + prob3 * ((po3 - m) / s) ** k
        + prob4 * ((po4 - m) / s) ** k
        + prob5 * ((po5 - m) / s) ** k
    )

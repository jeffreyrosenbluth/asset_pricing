import faicons as fa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

# Load data and compute static values
from shiny import reactive, render
from shiny.express import input, ui

sns.set_theme(style="whitegrid")

app_dir = Path(__file__).parent

# Add page title and sidebar
ui.page_opts(title="Modern Portfolio Theory", fillable=True)

with ui.sidebar(open="desktop"):
    ui.input_slider(
        "num_stocks",
        "Number of Stocks",
        min=2,
        max=50,
        value=25,
    )

    with ui.accordion(id="ret", open=False):
        with ui.accordion_panel("Returns"):
            ui.input_slider(
                "return1",
                "Return 1 ",
                min=1,
                max=50,
                value=10,
                post="%",
            )

            ui.input_slider(
                "return2",
                "Return 2",
                min=1,
                max=50,
                value=15,
                post="%",
            )

            ui.input_slider(
                "return3",
                "Return 3",
                min=1,
                max=50,
                value=20,
                post="%",
            )

        # with ui.accordion(id="sigma"):
        with ui.accordion_panel("Standard Deviations"):
            ui.input_slider(
                "std1",
                "Standard Deviation 1",
                min=1,
                max=50,
                value=15,
                post="%",
            )

            ui.input_slider(
                "std2",
                "Standard Deviation 2",
                min=1,
                max=50,
                value=25,
                post="%",
            )

            ui.input_slider(
                "std3",
                "Standard Deviation 3",
                min=1,
                max=50,
                value=35,
                post="%",
            )

        with ui.accordion_panel("Correlations"):
            ui.input_slider(
                "corr12",
                "Correlation 1-2",
                min=0,
                max=0.99,
                value=0.2,
            )

            ui.input_slider(
                "corr13",
                "Correlation 1-3",
                min=0,
                max=0.99,
                value=0.0,
            )

            ui.input_slider(
                "corr23",
                "Correlation 2-3",
                min=0,
                max=0.99,
                value=0.3,
            )

with ui.navset_tab(id="tab"):
    with ui.nav_panel("Diversification"):
        with ui.layout_columns(col_widths=[6, 6]):
            with ui.card(full_screen=True):
                ui.card_header("Diversifiacton")

                @render.plot
                def diverse():
                    plot_diversification(0.1, 0, 0.25, 0.5, input.num_stocks())

            with ui.card(full_screen=True):
                with ui.card_header(
                    class_="d-flex justify-content-between align-items-center"
                ):
                    "2 Risky Assets"

                @render.plot
                def asset2():
                    means = np.array([[input.return1() / 100, input.return2() / 100]])
                    plot_risky(means, cov2())

    with ui.nav_panel("Efficient Frontier"):
        with ui.layout_columns(col_widths=[6, 6]):
            with ui.card(full_screen=True):
                with ui.card_header(
                    class_="d-flex justify-content-between align-items-center"
                ):
                    "3 Risky Assets"

                @render.plot
                def asset3():
                    means = np.array(
                        [
                            [
                                input.return1() / 100,
                                input.return2() / 100,
                                input.return3() / 100,
                            ]
                        ]
                    )
                    plot_risky(means, cov3())

            with ui.card(full_screen=True):
                with ui.card_header(
                    class_="d-flex justify-content-between align-items-center"
                ):
                    "Frontier"

                @render.plot
                def frontier():
                    means = np.array(
                        [
                            [
                                input.return1() / 100,
                                input.return2() / 100,
                                input.return3() / 100,
                            ]
                        ]
                    )
                    plot_frontier(means, cov3(), 0.05)


ui.include_css(app_dir / "styles.css")

# --------------------------------------------------------
# Reactive calculations and effects
# --------------------------------------------------------


@reactive.calc
def cov2():
    covariance = input.corr12() * input.std1() / 100 * input.std2() / 100
    return np.array(
        [
            [(input.std1() / 100) ** 2, covariance],
            [covariance, (input.std2() / 100) ** 2],
        ]
    )


@reactive.calc
def cov3():
    covariance12 = input.corr12() * input.std1() / 100 * input.std2() / 100
    covariance13 = input.corr13() * input.std1() / 100 * input.std3() / 100
    covariance23 = input.corr23() * input.std2() / 100 * input.std3() / 100

    return np.array(
        [
            [(input.std1() / 100) ** 2, covariance12, covariance13],
            [covariance12, (input.std2() / 100) ** 2, covariance23],
            [covariance13, covariance23, (input.std3() / 100) ** 2],
        ]
    )


# --------------------------------------------------------
# Calculations and plotting
# --------------------------------------------------------
def diversification(std, corr, N):
    ns = np.arange(1, N + 1)
    vs = ns * std * std
    cs = ns * (ns - 1) * corr * std * std
    xs = vs + cs
    return np.sqrt(xs) / (np.arange(1, N + 1))


def plot_diversification(std, corr1, corr2, corr3, N):
    xs = diversification(std, corr1, N)
    ys = diversification(std, corr2, N)
    zs = diversification(std, corr3, N)

    xs_label = f"Correlation = {corr1:.2f}"
    ys_label = f"Correlation = {corr2:.2f}"
    zs_label = f"Correlation = {corr3:.2f}"

    fix, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, N + 1), zs * 100, label=zs_label)
    ax.plot(np.arange(1, N + 1), ys * 100, label=ys_label)
    ax.plot(np.arange(1, N + 1), xs * 100, label=xs_label)
    ax.set_xlabel("Number of Stocks")
    ax.set_ylabel("Standard Deviation (%)")
    ax.legend()


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


def plot_risky(means, cov):
    A, B, C, D = frontier_constants(means, cov)
    returns = np.linspace(0, 0.25, 100)
    stds = frontier_std(A, B, C, D, returns)
    return_points = means.T
    std_points = np.sqrt(np.diag(cov))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(stds * 100, returns * 100)
    ax.plot(std_points * 100, return_points * 100, "ro", label="Risky Assets")
    ax.plot(np.sqrt(1 / C) * 100, 100 * A / C, "go", label="MVP")
    ax.set_xlabel("Standard Deviation (%)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Mean Variance Frontier")
    ax.legend()


def plot_frontier(means, cov, rf):
    A, B, C, D = frontier_constants(means, cov)
    risky_returns = np.linspace(0, 0.25, 100)
    stds = frontier_std(A, B, C, D, risky_returns)
    return_points = means.T
    std_points = np.sqrt(np.diag(cov))
    cml_returns = np.linspace(0, 0.25, 100)
    slope = slope_of_tangency_portfolio(means, cov, rf)
    cml_stds = rf + slope * cml_returns

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(stds * 100, risky_returns * 100)
    ax.plot(cml_returns * 100, cml_stds * 100, label="CML")
    if return_points.size > 1:
        ax.plot(std_points * 100, return_points * 100, "ro", label="Risky Assets")
        ax.plot(np.sqrt(1 / C) * 100, 100 * A / C, "go", label="MVP")
    ax.set_xlabel("Standard Deviation (%)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Mean Variance Frontier w/ Risk Free Asset")
    ax.legend()

import faicons as fa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from core import (
    plot_frontier,
    plot_diversification,
    tangency_point,
    plot_mv,
)


# Load data and compute static values
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plotly


sns.set_theme(style="whitegrid")

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Diversification",
        ui.layout_columns(
            ui.card(
                ui.card_header("Risk for Portfolio Size"),
                output_widget("diverse"),
                ui.card_footer(
                    ui.input_slider(
                        "num_stocks",
                        "Number of Stocks",
                        min=2,
                        max=30,
                        value=10,
                    )
                ),
                height="650px",
            ),
            ui.markdown(
                """
            #### Diversification
            ---
            The more stocks we add the more we reduce the risk of the portfolio.
            Unless the correlations is zero, the risk of the portfolio will never
            be zero but will asymptotically approach a minimum value.
                """
            ),
            col_widths=[8, 4],
        ),
    ),
    ui.nav_panel(
        "2 Stocks",
        ui.layout_columns(
            ui.card(
                ui.card_header("Mean Variance Frontier for 2 Stocks"),
                output_widget("asset2_rf"),
                ui.card_footer(
                    ui.input_switch(
                        id="cml2", label="Capital Market Line", value=False
                    ),
                ),
                full_screen=True,
            ),
            ui.markdown(
                """
            #### 2 Stocks
            ---
            *Returns*: 8%, 15%  
            *Standard Deviations*: 10%, 30%  
            *Correlation*: 0.0  

            ---
            Notice that both stocks lie on the minimum variance frontier, 
            this is due to the fact that there is a unique combination of 
            the two stock that make up each return. That is  𝑤2 = 1 − 𝑤1. 
            Since all possible returns are generated by a portfolio of these 
            two stocks and both stocks lie on the frontier, we trivially 
            obtain a separation theorem. That is every investor chooses a 
            portfolio that is a linear combination of two frontier portfolios. 
            We will see in the 𝑁 > 2  case that the stocks generally do not 
            lie on the frontier, but that the separation theorm is still true!
                """
            ),
            col_widths=[8, 4],
            height="650px",
        ),
    ),
    ui.nav_panel(
        "Build Frontier",
        ui.layout_columns(
            ui.card(
                ui.card_header("Frontiers for Small Portfolios"),
                output_widget("mvfs"),
                ui.card_footer(
                    ui.input_checkbox_group(
                        "portfolio_size",
                        "Portfolio Sizes",
                        {
                            2: "2",
                            3: "3",
                            4: "4",
                            5: "5",
                        },
                        selected=[2, 3, 4, 5],
                        inline=True,
                    ),
                ),
                height="700px",
            ),
            ui.markdown(
                """
            #### Building the Efficient Frontier from Smaller Portfolios
            ---
            First build the 2 stock portfolios, then the 3 stock portfolios,
            ..., until we build the frontier of the 5 stock portfolio.
                """
            ),
            col_widths=[8, 4],
            height="650",
        ),
    ),
    ui.nav_panel(
        "3 Stocks",
        ui.layout_columns(
            ui.card(
                ui.card_header("Mean Variance Frontier for 3 Stocks"),
                output_widget("asset3_rf"),
                ui.input_switch(id="cml3", label="Capital Market Line", value=True),
            ),
            ui.markdown(
                """
            #### 3 Stocks
            ---
            *Returns*: 10%, 17%, 20%  
            *Standard Deviations*: 15%, 25%, 35%  
            *Correlation*: 0.0  

            ---
            Now the stocks do not lie on the frontier, but the separation still holds.
            Any efficient portfolio can be constructed as a linear combination of the
            risk free asset and the tangency portfolio. The tangency portfolio (TP).
                """
            ),
            col_widths=[8, 4],
            height="650px",
        ),
    ),
    title="Modern Portfolio Theory",
)


def server(input, output, session):
    @render_plotly
    def diverse():
        return plot_diversification(0.1, 0, 0.25, 0.5, input.num_stocks())

    @render_plotly
    def asset2_rf():
        c = 0
        means = np.array(
            [
                [
                    8 / 100,
                    15 / 100,
                ]
            ]
        )
        cov = np.array(
            [
                [(10 / 100) ** 2, c],
                [c, (30 / 100) ** 2],
            ]
        )
        rf = 0.05 if input.cml2() else None
        return plot_frontier(means, cov, 0.05, 0.16, rf=rf)

    @render_plotly
    def asset3_rf():
        means = np.array(
            [
                [
                    10 / 100,
                    17 / 100,
                    0.2,
                ]
            ]
        )
        cov = np.array(
            [
                [(15 / 100) ** 2, 0, 0],
                [0, (25 / 100) ** 2, 0],
                [0, 0, (35 / 100) ** 2],
            ]
        )
        tangency_point(means, cov, 0.05)
        rf = 0.05 if input.cml3() else None
        return plot_frontier(means, cov, 0.05, 0.25, rf=rf)

    @render_plotly
    def mvfs():
        return plot_mv(input.portfolio_size())


app = App(app_ui, server)
# app_dir = Path(__file__).parent

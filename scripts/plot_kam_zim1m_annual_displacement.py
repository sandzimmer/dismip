
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(__file__).resolve().parents[1] / "dismip/dismip_data/displacements.csv"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dismip/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2008
END_YEAR = 2021

COUNTRIES = ["ETH", "SOM", "SDN"]
MODELS = ["KAM", "ZIM-1M"]

MODEL_COLORS = {
    "KAM": "#D99A32",       # warm amber
    "ZIM-1M": "#168C91",    # deep teal
}

# Plot settings
SHOW_UNCERTAINTY = True
UNCERTAINTY_ALPHA = 0.18
LINE_WIDTH = 2.2
MARKER_SIZE = 4

# Independent y-axis limits for all panels
SHARE_Y = False

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["modeler"] = df["modeler"].astype(str).str.strip()
df["country"] = df["country"].astype(str).str.strip()

df["model_specification"] = (
    df["model_specification"]
    .astype(str)
    .str.replace("'", "", regex=False)
    .str.strip()
)

df["year"] = pd.to_numeric(
    df["year"], errors="coerce"
)

df["displacements"] = pd.to_numeric(
    df["displacements"], errors="coerce"
)

df = df[
    df["modeler"].isin(MODELS)
    & df["country"].isin(COUNTRIES)
    & df["year"].between(START_YEAR, END_YEAR)
].copy()

df = df.dropna(
    subset=[
        "year",
        "displacements",
        "model_specification"
    ]
)

df["year"] = df["year"].astype(int)

# ============================================================
# AGGREGATE MODEL SPECIFICATIONS
# ============================================================

# ============================================================
# SPEARMAN RANK CORRELATION
# ============================================================

def calculate_spearman(country):

    subset = summary[
        (summary["country"] == country)
        & (summary["modeler"].isin(["KAM", "ZIM-1M"]))
    ]

    # Align annual mean estimates by year
    comparison = subset.pivot_table(
        index="year",
        columns="modeler",
        values="mean",
        aggfunc="first"
    )

    comparison = comparison.reindex(
        columns=["KAM", "ZIM-1M"]
    ).dropna()

    if len(comparison) < 3:
        return np.nan, np.nan, len(comparison)

    if (
        comparison["KAM"].nunique() < 2
        or comparison["ZIM-1M"].nunique() < 2
    ):
        return np.nan, np.nan, len(comparison)

    rho, p_value = spearmanr(
        comparison["KAM"],
        comparison["ZIM-1M"]
    )

    return rho, p_value, len(comparison)

# First aggregate within each country, year, model and
# model specification. This prevents duplicate rows
# from being treated as independent model realizations.

country_specs = (
    df.groupby(
        [
            "modeler",
            "model_specification",
            "country",
            "year"
        ],
        as_index=False
    )["displacements"]
    .sum()
)

# ALL panel:
# Sum ETH + SOM + SDN for each specification and year
# BEFORE calculating mean/min/max.

all_specs = (
    country_specs.groupby(
        [
            "modeler",
            "model_specification",
            "year"
        ],
        as_index=False
    )["displacements"]
    .sum()
)

all_specs["country"] = "ALL"

# Combine ALL and country-level data
combined = pd.concat(
    [country_specs, all_specs],
    ignore_index=True
)

# Calculate mean and uncertainty range across specifications
summary = (
    combined.groupby(
        ["country", "modeler", "year"],
        as_index=False
    )["displacements"]
    .agg(
        mean="mean",
        minimum="min",
        maximum="max"
    )
)

# ============================================================
# FORMATTING
# ============================================================

def format_millions(x, pos):
    return f"{x / 1e6:g}"


def style_axis(ax, country, log_scale=False):

    ax.set_title(
        country,
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    ax.set_xlim(
        START_YEAR - 0.3,
        END_YEAR + 0.3
    )

    ax.set_xticks(
        np.arange(2008, END_YEAR + 1, 2)
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
        direction="out"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.7
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log_scale:
        ax.set_yscale("log")

        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10)
        )

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda value, pos:
                f"{value / 1e6:g}M"
                if value >= 1e6
                else f"{value:g}"
            )
        )

    else:
        ax.set_ylim(bottom=0)

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(format_millions)
        )


# ============================================================
# PLOT MODEL DATA
# ============================================================

def plot_country(ax, country, log_scale=False):

    subset = summary[
        summary["country"] == country
    ].copy()

    for model in MODELS:

        model_data = (
            subset[
                subset["modeler"] == model
            ]
            .sort_values("year")
        )

        if model_data.empty:
            continue

        years = model_data["year"].to_numpy()
        mean = model_data["mean"].to_numpy()
        lower = model_data["minimum"].to_numpy()
        upper = model_data["maximum"].to_numpy()

        # Shift zero values for logarithmic plotting
        if log_scale:
            mean = mean + 1
            lower = lower + 1
            upper = upper + 1

        color = MODEL_COLORS[model]

        ax.plot(
            years,
            mean,
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=model,
            zorder=3
        )

        if SHOW_UNCERTAINTY:
            ax.fill_between(
                years,
                lower,
                upper,
                color=color,
                alpha=UNCERTAINTY_ALPHA,
                linewidth=0,
                zorder=2
            )

    style_axis(
        ax,
        country,
        log_scale=log_scale
    )
    # --------------------------------------------------------
    # Annotate Spearman correlation
    # --------------------------------------------------------

    rho, p_value, n = calculate_spearman(country)

    if np.isfinite(rho):

        if p_value < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_value:.3f}"

        annotation = (
            f"Spearman ρ = {rho:.2f}\n"
            f"{p_text}, n = {n}"
        )

    else:
        annotation = "Spearman ρ = N/A"

    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#333333",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            boxstyle="round,pad=0.35"
        ),
        zorder=10
    )

    print(
        f"{country}: Spearman rho = {rho:.3f}, "
        f"p = {p_value:.4g}, n = {n}"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

def create_figure(log_scale=False):

    fig = plt.figure(
        figsize=(13.5, 8.5)
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.35, 1],
        hspace=0.43,
        wspace=0.32
    )

    # Large top panel spanning all three columns
    ax_all = fig.add_subplot(
        gs[0, :]
    )

    # Three smaller bottom panels
    ax_eth = fig.add_subplot(
        gs[1, 0],
        sharey=ax_all if SHARE_Y else None
    )

    ax_som = fig.add_subplot(
        gs[1, 1],
        sharey=ax_all if SHARE_Y else None
    )

    ax_sdn = fig.add_subplot(
        gs[1, 2],
        sharey=ax_all if SHARE_Y else None
    )

    axes = {
        "ALL": ax_all,
        "ETH": ax_eth,
        "SOM": ax_som,
        "SDN": ax_sdn
    }

    for country, ax in axes.items():
        plot_country(
            ax,
            country,
            log_scale=log_scale
        )

    # Axis labels
    ax_all.set_ylabel(
        "Displacements [million]",
        fontsize=12
    )

    ax_eth.set_ylabel(
        "Displacements [million]",
        fontsize=12
    )

    for ax in [ax_eth, ax_som, ax_sdn]:
        ax.set_xlabel(
            "Year",
            fontsize=11
        )

    # Shared legend below the panels
    handles = [
        Line2D(
            [0], [0],
            color=MODEL_COLORS[model],
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=model
        )
        for model in MODELS
    ]

    fig.legend(
        handles=handles,
        labels=MODELS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=12
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.95,
        bottom=0.13
    )

    suffix = "_log" if log_scale else ""

    filename = (
        f"annual_change_1m_four_panel{suffix}"
    )

    for extension in ["png", "pdf"]:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)

    print(f"Saved: {filename}.png / .pdf")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nModel specifications per model:")
    print(
        df.groupby("modeler")[
            "model_specification"
        ].nunique()
    )

    create_figure(log_scale=False)
    create_figure(log_scale=True)

    print("\nFinished.")

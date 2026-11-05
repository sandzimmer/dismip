#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# SETTINGS
# ============================================================

WD = Path(__file__).resolve().parents[1]

INPUT_DIR = WD / "dismip/dismip_data"
OUTPUT_DIR = WD / "dismip/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RIE_FILE = INPUT_DIR / "dismip_riedel.csv"
IDMC_FILE = WD / "data/idmc_fluvial_flood.csv"

COUNTRIES = ["ETH", "SOM", "SDN"]
PANEL_ORDER = ["ALL", "ETH", "SOM", "SDN"]

MODELS = ["RIE-0", "RIE-FLO"]

START_YEAR = 2008
END_YEAR = 2023
YEARS = np.arange(START_YEAR, END_YEAR + 1)

PERIODS = {
    "In-sample (2008–2021)": (2008, 2021),
    "Out-of-sample (2022–2023)": (2022, 2023),
}

COLORS = {
    "RIE-0": "#D99A32",       # Amber
    "RIE-FLO": "#168C91",     # Teal
    "IDMC": "#66023c",        # Burgundy
}

# Missing IDMC country-year combinations are always set to zero.
#
# Missing RIE model years are NOT automatically set to zero.
# Set True only if missing model predictions represent zero impact.
FILL_MISSING_MODEL_YEARS = False

SHOW_UNCERTAINTY = True
UNCERTAINTY_ALPHA = 0.18

# ============================================================
# LOAD RIE DATA
# ============================================================

rie = pd.read_csv(RIE_FILE)

rie = rie.rename(
    columns={
        "Year": "year",
        "Country": "country",
        "Impact": "displacements",
    }
)

rie["modeler"] = np.where(
    rie["Protection"].eq("No Protection"),
    "RIE-0",
    "RIE-FLO",
)

rie["model_specification"] = (
    rie["Protection"].astype(str)
    + "+"
    + rie["Impact Func ID"].astype(str)
)

rie["country"] = rie["country"].astype(str).str.strip()
rie["year"] = pd.to_numeric(
    rie["year"], errors="coerce"
)
rie["displacements"] = pd.to_numeric(
    rie["displacements"], errors="coerce"
)

rie = rie[
    rie["country"].isin(COUNTRIES)
    & rie["year"].between(START_YEAR, END_YEAR)
].dropna(
    subset=["year", "displacements"]
).copy()

rie["year"] = rie["year"].astype(int)

# Aggregate records within each model specification,
# country and year.
rie_country = (
    rie.groupby(
        [
            "modeler",
            "model_specification",
            "country",
            "year",
        ],
        as_index=False
    )["displacements"]
    .sum()
)

# ============================================================
# LOAD IDMC DATA
# ============================================================

idmc = pd.read_csv(IDMC_FILE)

idmc = idmc.rename(
    columns={
        "ISO3": "country",
        "Year": "year",
        "Disaster Internal Displacements": "displacements",
    }
)

idmc["country"] = idmc["country"].astype(str).str.strip()

idmc["year"] = pd.to_numeric(
    idmc["year"], errors="coerce"
)

idmc["displacements"] = pd.to_numeric(
    idmc["displacements"], errors="coerce"
)

idmc = idmc[
    idmc["country"].isin(COUNTRIES)
    & idmc["year"].between(START_YEAR, END_YEAR)
].dropna(
    subset=["year", "displacements"]
).copy()

idmc["year"] = idmc["year"].astype(int)

# ============================================================
# IDMC: INCLUDE ZERO-DISPLACEMENT YEARS
# ============================================================

# Aggregate all reported displacements per country-year.
idmc_country = (
    idmc.groupby(
        ["country", "year"],
        as_index=False
    )["displacements"]
    .sum()
)

# Create every country-year combination from 2008 to 2023.
full_index = pd.MultiIndex.from_product(
    [COUNTRIES, YEARS],
    names=["country", "year"]
)

# Replace missing IDMC country-year combinations with zero.
idmc_country = (
    idmc_country
    .set_index(["country", "year"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

print("\nIDMC annual displacement totals (zero years included):")
print(
    idmc_country.pivot(
        index="year",
        columns="country",
        values="displacements"
    ).to_string()
)

# ============================================================
# ADD ALL-COUNTRIES AGGREGATES
# ============================================================

def add_all_rie(data):

    all_data = (
        data.groupby(
            [
                "modeler",
                "model_specification",
                "year",
            ],
            as_index=False
        )["displacements"]
        .sum()
    )

    all_data["country"] = "ALL"

    return pd.concat(
        [data, all_data],
        ignore_index=True
    )


def add_all_idmc(data):

    all_data = (
        data.groupby(
            "year",
            as_index=False
        )["displacements"]
        .sum()
    )

    all_data["country"] = "ALL"

    return pd.concat(
        [data, all_data],
        ignore_index=True
    )


rie_all = add_all_rie(rie_country)
idmc_all = add_all_idmc(idmc_country)

# ============================================================
# ANNUAL MODEL SUMMARY
# ============================================================

annual_summary = (
    rie_all.groupby(
        ["country", "modeler", "year"],
        as_index=False
    )["displacements"]
    .agg(
        mean="mean",
        minimum="min",
        maximum="max",
    )
)

# ============================================================
# FOUR-PANEL FIGURE LAYOUT
# ============================================================

def make_axes(figsize=(12, 8)):

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.35, 1],
        hspace=0.42,
        wspace=0.30,
    )

    axes = {
        "ALL": fig.add_subplot(gs[0, :]),
        "ETH": fig.add_subplot(gs[1, 0]),
        "SOM": fig.add_subplot(gs[1, 1]),
        "SDN": fig.add_subplot(gs[1, 2]),
    }

    return fig, axes


def style_axis(ax, country):

    ax.set_title(
        country,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        color="0.85",
        linewidth=0.7,
        alpha=0.7,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )


def save_figure(fig, filename):

    for extension in ["png", "pdf"]:

        fig.savefig(
            OUTPUT_DIR / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2,
        )

    plt.close(fig)

    print(f"Saved: {filename}.png and {filename}.pdf")


# ============================================================
# FIGURE 1: ANNUAL DISPLACEMENT TIME SERIES
# ============================================================

# ============================================================
# SPEARMAN CORRELATION: RIE MODELS VS IDMC
# ============================================================


# ============================================================
# ANNOTATE CORRELATIONS IN UPPER-LEFT CORNER
# ============================================================
from scipy.stats import spearmanr

# ============================================================
# SPEARMAN CORRELATION WITH SIGNIFICANCE STARS
# ============================================================

from scipy.stats import spearmanr


def calculate_spearman_rie(country, model, start, end):

    model_data = annual_summary[
        (annual_summary["country"] == country)
        & (annual_summary["modeler"] == model)
        & annual_summary["year"].between(start, end)
    ][["year", "mean"]].copy()

    idmc_data = idmc_all[
        (idmc_all["country"] == country)
        & idmc_all["year"].between(start, end)
    ][["year", "displacements"]].copy()

    comparison = pd.merge(
        model_data,
        idmc_data,
        on="year",
        how="inner"
    ).dropna()

    n = len(comparison)

    # At least 3 observations required for significance testing
    if n < 3:
        return np.nan, np.nan, n

    if (
        comparison["mean"].nunique() < 2
        or comparison["displacements"].nunique() < 2
    ):
        return np.nan, np.nan, n

    rho, p_value = spearmanr(
        comparison["mean"],
        comparison["displacements"]
    )

    return rho, p_value, n


def significance_stars(p_value):

    if not np.isfinite(p_value):
        return ""

    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


# ============================================================
# ANNOTATE SPEARMAN CORRELATIONS
# ============================================================

def annotate_spearman_rie(ax, country):

    periods = [
        ("In-sample (2008–2021)", 2008, 2021),
        ("Out-of-sample (2022–2023)", 2022, 2023),
    ]

    x_position = 0.025
    y_position = 0.97
    line_spacing = 0.062

    fontsize = 10 if country == "ALL" else 8.5

    for period_label, start, end in periods:

        ax.text(
            x_position,
            y_position,
            period_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color="#333333",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=1.5
            ),
            zorder=10
        )

        y_position -= line_spacing

        for model in MODELS:

            rho, p_value, n = calculate_spearman_rie(
                country,
                model,
                start,
                end
            )

            if np.isfinite(rho):

                stars = significance_stars(p_value)

                annotation = (
                    f"{model}: ρ = {rho:.2f}{stars}"
                )

            else:
                annotation = (
                    f"{model}: ρ = N/A"
                )

            ax.text(
                x_position + 0.015,
                y_position,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=fontsize,
                color=COLORS[model],
                fontweight="medium",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=1.5
                ),
                zorder=10
            )

            print(
                f"{country} | {period_label} | "
                f"{model}: rho={rho:.3f}, "
                f"p={p_value:.4g}, n={n}"
            )

            y_position -= line_spacing

        y_position -= line_spacing * 0.35

def plot_annual_displacements():

    fig, axes = make_axes()

    for country, ax in axes.items():

        # Plot RIE models
        for model in MODELS:

            data = annual_summary[
                (annual_summary["country"] == country)
                & (annual_summary["modeler"] == model)
            ].sort_values("year")

            if data.empty:
                continue

            years = data["year"].to_numpy()
            means = data["mean"].to_numpy()
            minimum = data["minimum"].to_numpy()
            maximum = data["maximum"].to_numpy()

            ax.plot(
                years,
                means,
                color=COLORS[model],
                linewidth=2.2,
                marker="o",
                markersize=4,
                zorder=3,
            )

            if SHOW_UNCERTAINTY:

                ax.fill_between(
                    years,
                    minimum,
                    maximum,
                    color=COLORS[model],
                    alpha=UNCERTAINTY_ALPHA,
                    linewidth=0,
                    zorder=2,
                )

        # Plot IDMC, including zero-displacement years
        obs = idmc_all[
            idmc_all["country"] == country
        ].sort_values("year")

        ax.plot(
            obs["year"],
            obs["displacements"],
            color=COLORS["IDMC"],
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=4,
            zorder=4,
        )

        # Boundary between in-sample and out-of-sample
        ax.axvline(
            2021.5,
            color="0.45",
            linestyle=":",
            linewidth=1.2,
        )

        ax.set_xlim(2007.7, 2023.3)
        ax.set_ylim(bottom=0)

        ax.set_xticks(
            np.arange(2008, 2024, 4)
        )

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda value, pos: f"{value / 1e6:g}"
            )
        )

        style_axis(ax, country)
        annotate_spearman_rie(ax, country)

    axes["ALL"].set_ylabel(
        "Displacements [million]",
        fontsize=12
    )

    axes["ETH"].set_ylabel(
        "Displacements [million]",
        fontsize=12
    )

    for country in COUNTRIES:
        axes[country].set_xlabel(
            "Year",
            fontsize=11
        )

    # Shared legend
    handles = [
        Line2D(
            [0], [0],
            color=COLORS[model],
            marker="o",
            linewidth=2.2,
            label=model,
        )
        for model in MODELS
    ]

    handles.append(
        Line2D(
            [0], [0],
            color=COLORS["IDMC"],
            linestyle="--",
            marker="o",
            linewidth=1.8,
            label="IDMC",
        )
    )

    handles.append(
        Line2D(
            [0], [0],
            color="0.45",
            linestyle=":",
            linewidth=1.2,
            label="Start of out-of-sample period",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.95,
        bottom=0.13,
    )

    save_figure(
        fig,
        "annual_change_rie_four_panel"
    )


# ============================================================
# CALCULATE AAD AND RELATIVE DIFFERENCES
# ============================================================

def calculate_aad_comparison():

    results = []

    for country in PANEL_ORDER:

        for period, (start, end) in PERIODS.items():

            years = np.arange(start, end + 1)
            n_years = len(years)

            # ------------------------------------------------
            # IDMC AAD INCLUDING ZERO-DISPLACEMENT YEARS
            # ------------------------------------------------

            obs = idmc_all[
                (idmc_all["country"] == country)
                & idmc_all["year"].between(start, end)
            ]

            # All years are present because missing
            # country-year combinations were set to zero.
            idmc_aad = (
                obs["displacements"].sum() / n_years
            )

            if idmc_aad <= 0:

                print(
                    f"WARNING: IDMC AAD is zero for "
                    f"{country}, {period}. "
                    "Relative difference cannot be calculated."
                )

                continue

            # ------------------------------------------------
            # MODEL AAD FOR EACH SPECIFICATION
            # ------------------------------------------------

            for model in MODELS:

                model_data = rie_all[
                    (rie_all["country"] == country)
                    & (rie_all["modeler"] == model)
                    & rie_all["year"].between(start, end)
                ]

                specification_aads = []

                for specification, group in model_data.groupby(
                    "model_specification"
                ):

                    annual = (
                        group.groupby("year")[
                            "displacements"
                        ]
                        .sum()
                        .reindex(years)
                    )

                    # Missing model years are handled separately
                    # from missing IDMC years.
                    if annual.isna().any():

                        if FILL_MISSING_MODEL_YEARS:
                            annual = annual.fillna(0)

                        else:
                            print(
                                "WARNING: Incomplete model "
                                f"specification {model}, "
                                f"{specification}, {country}, "
                                f"{period}. Skipping specification."
                            )
                            continue

                    specification_aads.append(
                        annual.sum() / n_years
                    )

                if not specification_aads:
                    continue

                specification_aads = np.asarray(
                    specification_aads,
                    dtype=float
                )

                # --------------------------------------------
                # RELATIVE DIFFERENCE TO IDMC AAD
                # --------------------------------------------

                relative_differences = (
                    100 * (
                        specification_aads / idmc_aad - 1
                    )
                )

                results.append({

                    "country": country,
                    "period": period,
                    "modeler": model,

                    "idmc_aad": idmc_aad,

                    "model_aad_mean": (
                        specification_aads.mean()
                    ),

                    "model_aad_min": (
                        specification_aads.min()
                    ),

                    "model_aad_max": (
                        specification_aads.max()
                    ),

                    "relative_difference_mean": (
                        relative_differences.mean()
                    ),

                    "relative_difference_min": (
                        relative_differences.min()
                    ),

                    "relative_difference_max": (
                        relative_differences.max()
                    ),

                    "n_specifications": len(
                        specification_aads
                    ),
                })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_DIR / "rie_aad_relative_difference.csv",
        index=False,
    )

    print("\nAAD comparison:")
    print(result.to_string(index=False))

    return result


# ============================================================
# FIGURE 2: RELATIVE DIFFERENCE TO IDMC AAD
# ============================================================

def plot_aad_relative_difference(results):

    fig, axes = make_axes()

    period_order = list(PERIODS.keys())

    x = np.arange(len(period_order))
    bar_width = 0.32

    for country, ax in axes.items():

        country_data = results[
            results["country"] == country
        ]

        for i, model in enumerate(MODELS):

            model_data = (
                country_data[
                    country_data["modeler"] == model
                ]
                .set_index("period")
                .reindex(period_order)
            )

            means = model_data[
                "relative_difference_mean"
            ].to_numpy(dtype=float)

            minimum = model_data[
                "relative_difference_min"
            ].to_numpy(dtype=float)

            maximum = model_data[
                "relative_difference_max"
            ].to_numpy(dtype=float)

            positions = (
                x + (i - 0.5) * bar_width
            )

            valid = (
                np.isfinite(means)
                & np.isfinite(minimum)
                & np.isfinite(maximum)
            )

            if not valid.any():
                continue

            # Bars
            ax.bar(
                positions[valid],
                means[valid],
                width=bar_width,
                color=COLORS[model],
                alpha=0.90,
                zorder=3,
            )

            # Minimum-maximum uncertainty range
            lower_errors = np.maximum(
                means[valid] - minimum[valid],
                0
            )

            upper_errors = np.maximum(
                maximum[valid] - means[valid],
                0
            )

            ax.errorbar(
                positions[valid],
                means[valid],
                yerr=[
                    lower_errors,
                    upper_errors,
                ],
                fmt="none",
                color="black",
                linewidth=1.1,
                capsize=4,
                zorder=4,
            )

        # Zero = perfect agreement with IDMC AAD
        ax.axhline(
            0,
            color="0.25",
            linewidth=1.0,
            zorder=2,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                "In-sample\n2008–2021",
                "Out-of-sample\n2022–2023",
            ],
            fontsize=10,
        )

        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=100)
        )

        style_axis(ax, country)



    axes["ALL"].set_ylabel(
        "Relative difference to IDMC AAD [%]",
        fontsize=12
    )

    axes["ETH"].set_ylabel(
        "Relative difference [%]",
        fontsize=12
    )

    handles = [
        Patch(
            facecolor=COLORS[model],
            label=model,
        )
        for model in MODELS
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.95,
        bottom=0.14,
    )

    save_figure(
        fig,
        "rie_aad_relative_difference_four_panel"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nCreating annual displacement figure...")
    plot_annual_displacements()

    print("\nCalculating AAD comparisons...")
    aad_results = calculate_aad_comparison()

    if not aad_results.empty:

        print("\nCreating relative AAD difference figure...")
        plot_aad_relative_difference(aad_results)

    print("\nScript complete.")
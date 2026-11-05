
#!/usr/bin/env python3
"""
Four-panel comparison of annual average exposure (AAE)
and population across flood-displacement models.

(a) All countries combined
(b) Ethiopia
(c) Somalia
(d) Sudan

Each panel is split into event-based and probabilistic models.

Two versions:
1. Logarithmic: shared y-axis for population and AAE.
2. Linear: separate population and AAE axes, with
   country-specific limits.

AAE: narrow, solid bars.
Population: wider, semi-transparent background bars.

Error bars indicate minimum–maximum ranges.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.ticker import (
    FuncFormatter,
    FixedLocator,
    MaxNLocator,
    NullFormatter
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "dismip/dismip_data"
OUTPUT = ROOT / "dismip/output"

AAE_FILE = INPUT / "aae.csv"
POP_FILE = INPUT / "pop.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

COUNTRIES = {
    "ETH": "Ethiopia",
    "SOM": "Somalia",
    "SDN": "Sudan"
}

GROUPS = {
    "Event-based": [
        "KAM",
        "ZIM",
        "RIE-0",
        "RIE-FLO"
    ],
    "Probabilistic": [
        "MEI",
        "CIMA"
    ]
}

LABELS = {
    "KAM": "KAM",
    "ZIM": "ZIM (1M/D)",
    "RIE-0": "RIE-0",
    "RIE-FLO": "RIE-FLO",
    "MEI": "MEI (a/b)",
    "CIMA": "CIMA"
}

COLORS = {
    "KAM": "#6d9e94",
    "ZIM": "#6d9e94",
    "RIE-0": "#7393be",
    "RIE-FLO": "#7393be",
    "MEI": "#c48d62",
    "CIMA": "#9480ae"
}

POP_WIDTH = 0.54
AAE_WIDTH = 0.32
BAR_STEP = 0.70

POP_ALPHA = 0.30

FIG_WIDTH = 13
FIG_HEIGHT = 8

# Logarithmic axis settings
LOG_YMAX = 2e8
LOG_TICKS = [1e4, 1e5, 1e6, 1e7, 1e8]

# Linear axis padding above highest value
LINEAR_PADDING = 1.15

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ============================================================
# STANDARDIZE MODEL NAMES
# ============================================================

def standardize_models(df):

    df = df.copy()

    mapping = {
        "ZIM-1M": "ZIM",
        "ZIM-D": "ZIM",
        "MEI-CAPRA": "MEI",
        "MEI-IVM": "MEI"
    }

    df["family"] = df["modeler"].replace(mapping)

    return df


# ============================================================
# LOAD AAE
# ============================================================

def load_aae():

    df = pd.read_csv(AAE_FILE)

    required = [
        "country",
        "modeler",
        "model_specification",
        "aae",
        "year"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing AAE columns: {missing}"
        )

    df = df[required].drop_duplicates()

    df = df[
        df["country"].isin(COUNTRIES)
    ].copy()

    df["aae"] = pd.to_numeric(
        df["aae"],
        errors="coerce"
    )

    df = df.dropna(subset=["aae"])

    df = standardize_models(df)

    return df


# ============================================================
# LOAD POPULATION
# ============================================================

def load_population():

    df = pd.read_csv(POP_FILE)

    required = [
        "country",
        "modeler",
        "pop",
        "year",
        "SSP"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing population columns: {missing}"
        )

    df = df[required].drop_duplicates()

    df = df[
        df["country"].isin(COUNTRIES)
    ].copy()

    df["pop"] = pd.to_numeric(
        df["pop"],
        errors="coerce"
    )

    df = df.dropna(subset=["pop"])

    df = standardize_models(df)

    return df


# ============================================================
# AAE SUMMARY
# ============================================================

def summarize_aae(df, country=None):

    data = df.copy()

    if country is not None:
        data = data[
            data["country"] == country
        ]

    # Aggregate across countries for each model,
    # specification and year.

    grouped = (
        data.groupby(
            [
                "family",
                "modeler",
                "model_specification",
                "year"
            ],
            dropna=False
        )["aae"]
        .sum()
        .reset_index()
    )

    # Average across years within each specification.

    spec_means = (
        grouped.groupby(
            [
                "family",
                "modeler",
                "model_specification"
            ],
            dropna=False
        )["aae"]
        .mean()
        .reset_index()
    )

    # Mean and range across specifications.

    stats = (
        spec_means.groupby("family")["aae"]
        .agg(["mean", "min", "max"])
    )

    return stats


# ============================================================
# POPULATION SUMMARY
# ============================================================

def summarize_population(df, country=None):

    data = df.copy()

    if country is not None:
        data = data[
            data["country"] == country
        ]

    # Sum population across countries for each
    # model, year and SSP.

    grouped = (
        data.groupby(
            [
                "family",
                "modeler",
                "SSP",
                "year"
            ],
            dropna=False
        )["pop"]
        .sum()
        .reset_index()
    )

    # Average across years within each SSP.

    scenario_means = (
        grouped.groupby(
            [
                "family",
                "modeler",
                "SSP"
            ],
            dropna=False
        )["pop"]
        .mean()
        .reset_index()
    )

    # Mean and range across scenarios.

    stats = (
        scenario_means.groupby("family")["pop"]
        .agg(["mean", "min", "max"])
    )

    return stats


# ============================================================
# Y-AXIS FORMATTING
# ============================================================

def format_y(value, _):

    if value >= 1e9:
        return f"{value / 1e9:g}B"

    if value >= 1e6:
        return f"{value / 1e6:g}M"

    if value >= 1e3:
        return f"{value / 1e3:g}k"

    return f"{value:g}"


# ============================================================
# DRAW INDIVIDUAL SUBPLOT
# ============================================================

def draw_panel(
    ax,
    aae_stats,
    pop_stats,
    group,
    show_title=False,
    ax_aae=None
):

    # Log version: both datasets use the same axis.
    # Linear version: AAE uses a secondary axis.

    if ax_aae is None:
        ax_aae = ax

    models = GROUPS[group]

    positions = (
        np.arange(len(models)) * BAR_STEP
    )

    for x, model in zip(positions, models):

        color = COLORS[model]

        # ----------------------------------------------------
        # POPULATION (background)
        # ----------------------------------------------------

        if model in pop_stats.index:

            row = pop_stats.loc[model]

            mean = float(row["mean"])
            low = float(row["min"])
            high = float(row["max"])

            if np.isfinite(mean) and mean > 0:

                ax.bar(
                    x,
                    mean,
                    width=POP_WIDTH,
                    color=color,
                    alpha=POP_ALPHA,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=2
                )

                if (
                    np.isfinite(low)
                    and np.isfinite(high)
                    and low >= 0
                    and high >= mean >= low
                ):

                    ax.errorbar(
                        x,
                        mean,
                        yerr=[
                            [mean - low],
                            [high - mean]
                        ],
                        fmt="none",
                        ecolor="#777777",
                        elinewidth=1.0,
                        capsize=2.5,
                        capthick=1.0,
                        alpha=0.65,
                        zorder=3
                    )

        # ----------------------------------------------------
        # AAE (foreground)
        # ----------------------------------------------------

        if model in aae_stats.index:

            row = aae_stats.loc[model]

            mean = float(row["mean"])
            low = float(row["min"])
            high = float(row["max"])

            if np.isfinite(mean) and mean > 0:

                ax_aae.bar(
                    x,
                    mean,
                    width=AAE_WIDTH,
                    color=color,
                    alpha=1.0,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4
                )

                if (
                    np.isfinite(low)
                    and np.isfinite(high)
                    and low >= 0
                    and high >= mean >= low
                ):

                    ax_aae.errorbar(
                        x,
                        mean,
                        yerr=[
                            [mean - low],
                            [high - mean]
                        ],
                        fmt="none",
                        ecolor="#303030",
                        elinewidth=1.2,
                        capsize=3,
                        capthick=1.2,
                        zorder=5
                    )

    # --------------------------------------------------------
    # AXIS FORMATTING
    # --------------------------------------------------------

    ax.set_xticks(
        positions,
        [LABELS[m] for m in models]
    )

    ax.tick_params(
        axis="x",
        rotation=48,
        length=0,
        pad=5,
        labelsize=12
    )

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")

    ax.set_xlim(
        positions.min() - 0.45,
        positions.max() + 0.45
    )

    ax.grid(
        axis="y",
        which="major",
        alpha=0.15,
        zorder=0
    )

    ax.set_axisbelow(True)

    if show_title:
        ax.set_title(
            group,
            weight="semibold",
            pad=11
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ============================================================
# CALCULATE LOGARITHMIC LIMITS
# ============================================================

def calculate_log_limits(panel_stats):

    all_values = []

    for aae_stats, pop_stats in panel_stats.values():

        for stats in [aae_stats, pop_stats]:

            values = stats[
                ["min", "max"]
            ].to_numpy().ravel()

            values = values[
                np.isfinite(values)
                & (values > 0)
            ]

            all_values.extend(values)

    all_values = np.asarray(all_values)

    if len(all_values) == 0:
        raise ValueError(
            "No positive AAE or population values found."
        )

    ymin = 10 ** np.floor(
        np.log10(all_values.min())
    )

    ymax = max(
        LOG_YMAX,
        all_values.max() * 1.05
    )

    return ymin, ymax


# ============================================================
# CALCULATE COUNTRY-SPECIFIC LINEAR LIMITS
# ============================================================

def calculate_linear_limits(aae_stats, pop_stats):

    pop_values = pop_stats[
        ["min", "max"]
    ].to_numpy().ravel()

    aae_values = aae_stats[
        ["min", "max"]
    ].to_numpy().ravel()

    pop_values = pop_values[
        np.isfinite(pop_values)
    ]

    aae_values = aae_values[
        np.isfinite(aae_values)
    ]

    pop_ymax = (
        pop_values.max() * LINEAR_PADDING
        if len(pop_values) > 0 else 1
    )

    aae_ymax = (
        aae_values.max() * LINEAR_PADDING
        if len(aae_values) > 0 else 1
    )

    pop_ymax = max(pop_ymax, 1)
    aae_ymax = max(aae_ymax, 1)

    return (0, pop_ymax), (0, aae_ymax)


# ============================================================
# CREATE FIGURE
# ============================================================

def make_figure(panel_stats, scale="log"):

    if scale not in ("log", "linear"):
        raise ValueError("scale must be 'log' or 'linear'")

    # Log version uses identical limits everywhere.
    if scale == "log":

        ymin, ymax = calculate_log_limits(panel_stats)

        print(
            f"\nShared logarithmic limits: "
            f"{ymin:,.0f} – {ymax:,.0f}"
        )

    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.25, 1],
        hspace=0.58,
        wspace=0.45 if scale == "linear" else 0.18
    )

    panels = [
        (
            gs[0, :],
            "ALL",
            "a",
            "All countries"
        ),
        (
            gs[1, 0],
            "ETH",
            "b",
            "Ethiopia"
        ),
        (
            gs[1, 1],
            "SOM",
            "c",
            "Somalia"
        ),
        (
            gs[1, 2],
            "SDN",
            "d",
            "Sudan"
        )
    ]

    for slot, country, letter, name in panels:

        sub = slot.subgridspec(
            1,
            2,
            width_ratios=[5.2, 3.3],
            wspace=0.06
        )

        # Independent primary axes.
        ax_event = fig.add_subplot(
            sub[0, 0]
        )

        ax_prob = fig.add_subplot(
            sub[0, 1]
        )

        aae_stats, pop_stats = panel_stats[country]

        # ----------------------------------------------------
        # CALCULATE AXIS LIMITS
        # ----------------------------------------------------

        if scale == "linear":

            (pop_ymin, pop_ymax), (
                aae_ymin, aae_ymax
            ) = calculate_linear_limits(
                aae_stats,
                pop_stats
            )

            print(
                f"\n{country} linear limits:"
                f"\n  Population: "
                f"{pop_ymin:,.0f} – {pop_ymax:,.0f}"
                f"\n  AAE: "
                f"{aae_ymin:,.0f} – {aae_ymax:,.0f}"
            )

        else:

            pop_ymin, pop_ymax = ymin, ymax
            aae_ymin, aae_ymax = ymin, ymax

        # ----------------------------------------------------
        # SECONDARY AXES FOR LINEAR VERSION
        # ----------------------------------------------------

        if scale == "linear":

            ax_event_aae = ax_event.twinx()
            ax_prob_aae = ax_prob.twinx()

            for primary, secondary in [
                (ax_event, ax_event_aae),
                (ax_prob, ax_prob_aae)
            ]:

                # Transparent secondary axis so the
                # background population bars remain visible.
                secondary.patch.set_visible(False)

                secondary.spines["top"].set_visible(False)

                # Draw AAE bars above population bars.
                secondary.set_zorder(
                    primary.get_zorder() + 1
                )

        else:

            ax_event_aae = ax_event
            ax_prob_aae = ax_prob

        # ----------------------------------------------------
        # DRAW PANELS
        # ----------------------------------------------------

        draw_panel(
            ax_event,
            aae_stats,
            pop_stats,
            "Event-based",
            show_title=(country == "ALL"),
            ax_aae=ax_event_aae
        )

        draw_panel(
            ax_prob,
            aae_stats,
            pop_stats,
            "Probabilistic",
            show_title=(country == "ALL"),
            ax_aae=ax_prob_aae
        )

        # ----------------------------------------------------
        # AXIS SCALES AND TICKS
        # ----------------------------------------------------

        if scale == "log":

            for ax in [ax_event, ax_prob]:

                ax.set_yscale("log")
                ax.set_ylim(ymin, ymax)

                # Explicit ticks including 1M and 100M.
                ax.yaxis.set_major_locator(
                    FixedLocator(LOG_TICKS)
                )

                ax.yaxis.set_major_formatter(
                    FuncFormatter(format_y)
                )

                ax.yaxis.set_minor_formatter(
                    NullFormatter()
                )

        else:

            # Population: left axis
            for ax in [ax_event, ax_prob]:

                ax.set_yscale("linear")
                ax.set_ylim(pop_ymin, pop_ymax)

                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=5)
                )

                ax.yaxis.set_major_formatter(
                    FuncFormatter(format_y)
                )

            # AAE: right axis
            for ax in [ax_event_aae, ax_prob_aae]:

                ax.set_yscale("linear")
                ax.set_ylim(aae_ymin, aae_ymax)

                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=5)
                )

                ax.yaxis.set_major_formatter(
                    FuncFormatter(format_y)
                )

        # ----------------------------------------------------
        # Y-AXIS LABELS
        # ----------------------------------------------------

        if country in ("ALL", "ETH"):

            ax_event.set_ylabel(
                "Population" if scale == "linear"
                else "Number of people"
            )

        else:

            ax_event.tick_params(
                axis="y",
                left=False,
                labelleft=False
            )

        # Hide internal population labels.
        ax_prob.tick_params(
            axis="y",
            left=False,
            labelleft=False
        )

        if scale == "linear":

            # Hide internal secondary-axis labels.
            ax_event_aae.tick_params(
                axis="y",
                right=False,
                labelright=False
            )

            # Display AAE ticks on the right side
            # of each country panel.
            ax_prob_aae.tick_params(
                axis="y",
                right=True,
                labelright=True
            )

            ax_prob_aae.set_ylabel(
                "Annual average exposure (AAE)"
            )

        # ----------------------------------------------------
        # PANEL TITLES
        # ----------------------------------------------------

        ax_event.text(
            0,
            1.15,
            f"({letter})  {name}",
            transform=ax_event.transAxes,
            fontsize=16,
            weight="bold",
            va="bottom"
        )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend = [
        Patch(
            facecolor="#7393be",
            edgecolor="white",
            alpha=POP_ALPHA,
            label="Population"
        ),
        Patch(
            facecolor="#7393be",
            edgecolor="white",
            alpha=1.0,
            label="Annual average exposure (AAE)"
        )
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        fontsize=12,
        columnspacing=2,
        handlelength=2
    )

    fig.subplots_adjust(
        left=0.08 if scale == "linear" else 0.075,
        right=0.93 if scale == "linear" else 0.985,
        top=0.91,
        bottom=0.19
    )

    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    for ext in ("png", "pdf"):

        filename = (
            OUTPUT /
            f"aae_population_grouped_4panel_{scale}.{ext}"
        )

        fig.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        print("Saved:", filename)

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    aae = load_aae()
    pop = load_population()

    panel_stats = {}

    for country in [None, "ETH", "SOM", "SDN"]:

        key = "ALL" if country is None else country

        aae_stats = summarize_aae(
            aae,
            country
        )

        pop_stats = summarize_population(
            pop,
            country
        )

        panel_stats[key] = (
            aae_stats,
            pop_stats
        )

        print(f"\n--- {key}: AAE ---")
        print(aae_stats.to_string())

        print(f"\n--- {key}: Population ---")
        print(pop_stats.to_string())

    # Generate logarithmic version.
    make_figure(
        panel_stats,
        scale="log"
    )

    # Generate linear version with country-specific limits.
    make_figure(
        panel_stats,
        scale="linear"
    )


if __name__ == "__main__":
    main()

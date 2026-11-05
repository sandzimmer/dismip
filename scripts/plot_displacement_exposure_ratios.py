#!/usr/bin/env python3
"""
DISMIP displacement / exposure fractions.

A4 landscape figure:
    All countries
    Ethiopia | Somalia | Sudan

Columns:
    Modelled | IDMC | DFO

Changes:
- Wider percentage bars
- Shared 0–100% scale for all bars
- Tick labels: 0, 25, 50, 75, 100%
- Numerical ranges only in All countries
- Ranges placed after the bars
- Country panels have no numerical range labels
- Values >100% marked with diamonds
- PDF and PNG output
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "dismip/dismip_data"
OUTPUT = ROOT / "dismip/output"

IDMC_FILE = (
    ROOT / "dismip/new_idmc_data/"
    "annual_country_displacements.csv"
)

DFO_FILE = (
    ROOT / "data/dfo_country_displacements.csv"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

COUNTRIES = ["ETH", "SOM", "SDN"]

STUDY_START = 2008
STUDY_END = 2021

DFO_FULL_START = None
DFO_FULL_END = None

GROUPS = {
    "Event-based": [
        ("KAM", "ZIM-1M"),
        ("ZIM-D",),
        ("RIE-0", "RIE-FLO"),
    ],
    "Probabilistic": [
        ("MEI-CAPRA", "MEI-IVM"),
        ("CIMA",),
    ],
}

LABELS = {
    "MEI-CAPRA": "MEI-a",
    "MEI-IVM": "MEI-b",
}

MODEL_ORDER = [
    model
    for families in GROUPS.values()
    for family in families
    for model in family
]

MODEL_GROUP = {
    model: group
    for group, families in GROUPS.items()
    for family in families
    for model in family
}

TYPES = [
    "Modelled",
    "IDMC",
    "DFO",
]

COLORS = {
    "Modelled": "#d39b37",
    "IDMC": "#873b59",
    "DFO": "#478c9b",
}

BACKGROUND = "#dfe3e7"
GRID_COLOR = "#e6e8eb"
TEXT_COLOR = "#30343b"


# ============================================================
# FIGURE SETTINGS
# ============================================================

# A4 landscape
FIGSIZE = (12, 9)

MODEL_FONT = 10
VALUE_FONT = 11
HEADING_FONT =12
GROUP_FONT = 11
PANEL_FONT = 15
TICK_FONT = 11

# Wider and thicker bars
BAR_HEIGHT = 0.04

# Fraction of each metric column occupied by bars.
# ALL reserves space for the numerical ranges.
BAR_FRACTION_ALL = 0.73

# Country panels use almost all available width.
BAR_FRACTION_COUNTRY = 0.96

# Percentage scale
TICKS = [0, 0.25, 0.50, 0.75, 1.0]
TICK_LABELS = ["0", "25", "50", "75", "100%"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# ============================================================
# DATA HELPERS
# ============================================================

def normalize(df):

    df = df.copy()

    df["country"] = (
        df["country"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"AB9": "SDN"})
    )

    df["model_specification"] = (
        df["model_specification"]
        .fillna("")
        .astype(str)
        .str.replace("'", "", regex=False)
        .str.strip()
    )

    return df[
        df.country.isin(COUNTRIES)
    ].copy()


# ============================================================
# EXPOSURE
# ============================================================

def prepare_exposure(aad, raw):

    base = normalize(raw)

    base["aae"] = pd.to_numeric(
        base["aae"],
        errors="coerce"
    )

    parts = [
        base[
            ~base.modeler.isin([
                "RIE-0",
                "RIE-FLO",
                "MEI",
                "ZIM",
            ])
        ].copy()
    ]

    # ZIM exposure
    zim = base[
        base.modeler.eq("ZIM")
    ].copy()

    # ZIM-1M retains its original exposure specifications.
    zim_1m = zim.copy()
    zim_1m["modeler"] = "ZIM-1M"
    parts.append(zim_1m)

    # New ZIM-D AAD specifications are mean/lower/upper.
    # These represent alternative displacement estimates, not
    # different exposure scenarios. Give each the same exposure
    # denominator for its country.
    zimd_specs = (
        aad.loc[aad.modeler.eq("ZIM-D"),
                ["country", "modeler", "model_specification"]]
        .drop_duplicates()
    )

    if not zimd_specs.empty:
        zim_exposure = zim[["country", "aae"]].dropna()
        variation = zim_exposure.groupby("country")["aae"].nunique()
        if variation.gt(1).any():
            warnings.warn(
                "ZIM has multiple exposure values per country; "
                "using their mean for the new ZIM-D ensemble. "
                "Check that this matches the intended H08/WaterGAP exposure."
            )
        zim_exposure = (
            zim_exposure.groupby("country", as_index=False)["aae"].mean()
        )
        parts.append(
            zimd_specs.merge(
                zim_exposure, on="country", how="left",
                validate="many_to_one"
            )
        )

    # MEI exposure
    mei = (
        base[base.modeler.eq("MEI")]
        [["country", "aae"]]
        .drop_duplicates()
    )

    if mei.groupby("country").size().gt(1).any():

        raise ValueError(
            "Multiple MEI exposure values per country."
        )

    for model in ["MEI-CAPRA", "MEI-IVM"]:

        specs = (
            aad[aad.modeler.eq(model)]
            [
                [
                    "country",
                    "modeler",
                    "model_specification",
                ]
            ]
            .drop_duplicates()
        )

        parts.append(
            specs.merge(
                mei,
                on="country",
                how="left",
                validate="many_to_one",
            )
        )

    # RIE exposure
    rie = base[
        base.modeler.isin([
            "RIE-0",
            "RIE-FLO",
        ])
    ].copy()

    for i in range(20):

        part = rie.copy()

        part["model_specification"] = (
            part["model_specification"]
            + "+"
            + str(i)
        )

        part["modeler"] = np.where(
            part.model_specification.str.startswith(
                "No Protection"
            ),
            "RIE-0",
            "RIE-FLO",
        )

        parts.append(part)

    exposure = pd.concat(
        parts,
        ignore_index=True
    )

    exposure = exposure[
        exposure.modeler.isin(MODEL_ORDER)
    ]

    keys = [
        "country",
        "modeler",
        "model_specification",
    ]

    exposure = (
        exposure[keys + ["aae"]]
        .drop_duplicates()
    )

    if exposure.duplicated(keys).any():

        raise ValueError(
            "Duplicate AAE specification keys."
        )

    return exposure


# ============================================================
# LOAD MODEL DATA
# ============================================================

def load_models():

    aad = normalize(
        pd.read_csv(INPUT / "aad.csv")
    )

    aae = pd.read_csv(
        INPUT / "aae.csv"
    )

    aad["aad"] = pd.to_numeric(
        aad["aad"],
        errors="coerce"
    )

    keys = [
        "country",
        "modeler",
        "model_specification",
    ]

    aad = aad[
        aad.modeler.isin(MODEL_ORDER)
    ][keys + ["aad"]]

    if aad.duplicated(keys).any():

        raise ValueError(
            "Duplicate model AAD keys."
        )

    exposure = prepare_exposure(
        aad,
        aae
    )

    df = aad.merge(
        exposure,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if df.aae.isna().any():

        warnings.warn(
            f"{df.aae.isna().sum()} rows "
            "lack matched exposure."
        )

        print(
            df.loc[
                df.aae.isna(),
                keys
            ].head(20)
        )

    # Diagnostic: show whether the new ZIM-D specifications matched.
    zimd_check = df[df.modeler.eq("ZIM-D")]
    print("\nZIM-D AAD / exposure merge:")
    print(zimd_check[keys + ["aad", "aae"]].to_string(index=False))

    df = df[
        df.aae.gt(0)
        & df.aad.notna()
    ].copy()

    if df.empty:
        raise ValueError("No valid model data.")

    return df


# ============================================================
# OBSERVATIONAL DATA
# ============================================================

def load_observations():

    dfo = pd.read_csv(DFO_FILE)

    allocations = [
        "Displaced_area",
        "Displaced_population",
        "Displaced_exposure",
    ]

    if all(c in dfo.columns for c in allocations):

        dfo["displaced"] = (
            dfo[allocations]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .mean(axis=1)
        )

    elif "Displaced" in dfo.columns:

        dfo["displaced"] = pd.to_numeric(
            dfo["Displaced"],
            errors="coerce"
        )

    else:

        raise ValueError(
            "DFO displacement columns missing."
        )

    country_col = next(
        (
            c for c in [
                "country",
                "Country",
                "ISO3",
            ]
            if c in dfo.columns
        ),
        None
    )

    if country_col is None:

        raise ValueError(
            "DFO country column missing."
        )

    dfo["country"] = (
        dfo[country_col]
        .astype(str)
        .str.strip()
        .replace({
            "Ethiopia": "ETH",
            "Somalia": "SOM",
            "Sudan": "SDN",
            "AB9": "SDN",
        })
    )

    dfo = dfo[
        dfo.country.isin(COUNTRIES)
    ].copy()

    numeric = pd.to_numeric(
        dfo["Began"],
        errors="coerce"
    )

    dates = pd.to_datetime(
        dfo["Began"],
        errors="coerce"
    ).dt.year

    dfo["year"] = numeric.where(
        numeric.between(1800, 2100),
        dates
    )

    dfo = dfo.dropna(
        subset=["year", "displaced"]
    ).copy()

    dfo["year"] = dfo.year.astype(int)

    if dfo.empty:

        raise ValueError(
            "No valid DFO observations."
        )

    dfo_start = int(
        DFO_FULL_START
        if DFO_FULL_START is not None
        else dfo.year.min()
    )

    dfo_end = int(
        DFO_FULL_END
        if DFO_FULL_END is not None
        else dfo.year.max()
    )

    if dfo_start > dfo_end:
        raise ValueError("Invalid DFO period.")

    # IDMC
    idmc = pd.read_csv(IDMC_FILE)

    idmc["country"] = (
        idmc["ISO3"]
        .astype(str)
        .str.strip()
        .replace({"AB9": "SDN"})
    )

    idmc = idmc[
        idmc.country.isin(COUNTRIES)
    ].copy()

    idmc["year"] = pd.to_numeric(
        idmc["Year"],
        errors="coerce"
    )

    idmc["displaced"] = pd.to_numeric(
        idmc["displacements_filtered"],
        errors="coerce"
    )

    idmc = idmc.dropna(
        subset=["year"]
    ).copy()

    idmc["year"] = idmc.year.astype(int)

    if idmc.empty:

        raise ValueError(
            "No valid IDMC observations."
        )

    idmc_start = int(idmc.year.min())
    idmc_end = int(idmc.year.max())

    def annual(df, start, end):

        return (
            df[
                df.year.between(start, end)
            ]
            .groupby("country")
            .displaced
            .sum()
            .reindex(
                COUNTRIES,
                fill_value=0
            )
            / (end - start + 1)
        )

    refs = {
        ("Event-based", "IDMC"): annual(
            idmc,
            STUDY_START,
            STUDY_END
        ),
        ("Event-based", "DFO"): annual(
            dfo,
            STUDY_START,
            STUDY_END
        ),
        ("Probabilistic", "IDMC"): annual(
            idmc,
            idmc_start,
            idmc_end
        ),
        ("Probabilistic", "DFO"): annual(
            dfo,
            dfo_start,
            dfo_end
        ),
    }

    periods = {
        "Event-based": (
            f"{STUDY_START}–{STUDY_END}"
        ),
        "Probabilistic": (
            f"IDMC {idmc_start}–{idmc_end}; "
            f"DFO {dfo_start}–{dfo_end}"
        ),
    }

    return refs, periods


# ============================================================
# CALCULATE FRACTIONS
# ============================================================

def calculate(models, refs):

    keys = [
        "modeler",
        "model_specification",
    ]

    complete = (
        models.groupby(keys)
        .country
        .nunique()
    )

    complete = complete[
        complete.eq(len(COUNTRIES))
    ].index

    if len(complete) == 0:

        raise ValueError(
            "No complete model specifications "
            "across all three countries."
        )

    pooled = (
        models.set_index(keys)
        .loc[complete]
        .reset_index()
        .groupby(
            keys,
            as_index=False
        )[["aad", "aae"]]
        .sum()
    )

    pooled["country"] = "ALL"

    data = pd.concat(
        [models, pooled],
        ignore_index=True
    )

    data["group"] = (
        data.modeler.map(MODEL_GROUP)
    )

    frames = []

    for kind in TYPES:

        df = data.copy()

        if kind == "Modelled":

            df["numerator"] = df.aad

        else:

            df["numerator"] = np.nan

            for group in GROUPS:

                reference = refs[(group, kind)]

                mask = df.group.eq(group)

                df.loc[
                    mask,
                    "numerator"
                ] = (
                    df.loc[
                        mask,
                        "country"
                    ]
                    .map(reference)
                )

                df.loc[
                    mask & df.country.eq("ALL"),
                    "numerator"
                ] = reference.sum()

        df["type"] = kind

        df["ratio"] = (
            df.numerator / df.aae
        )

        frames.append(
            df[
                [
                    "country",
                    "group",
                    "modeler",
                    "model_specification",
                    "type",
                    "ratio",
                ]
            ]
        )

    long = (
        pd.concat(
            frames,
            ignore_index=True
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(subset=["ratio"])
    )

    stats = (
        long.groupby(
            ["country", "group", "modeler", "type"]
        )
        .ratio
        .agg(median="median", minimum="min", maximum="max", n="size")
        .reset_index()
    )

    # The central ZIM-D estimate is explicitly 'mean'; do not
    # calculate its central bar from the median of CI endpoints.
    central = long[
        long.modeler.eq("ZIM-D")
        & long.model_specification.eq("mean")
    ][["country", "group", "modeler", "type", "ratio"]]
    if not central.empty:
        central = central.rename(columns={"ratio": "central_ratio"})
        stats = stats.merge(
            central,
            on=["country", "group", "modeler", "type"],
            how="left", validate="one_to_one"
        )
        mask = stats.modeler.eq("ZIM-D") & stats.central_ratio.notna()
        stats.loc[mask, "median"] = stats.loc[mask, "central_ratio"]
        stats = stats.drop(columns="central_ratio")

    return long, stats


# ============================================================
# FORMAT NUMERIC RANGES
# ============================================================

def percent(x):

    value = x * 100

    if abs(value) < 10:
        return f"{value:.1f}%"

    return f"{value:.0f}%"


def range_text(low, high):

    a = percent(low)
    b = percent(high)

    if (
        a == b
        or np.isclose(
            low,
            high,
            rtol=1e-6,
            atol=1e-10
        )
    ):
        return a

    return f"{a[:-1]}–{b}"


# ============================================================
# PLOTTING
# ============================================================

def plot(stats, periods):

    fig = plt.figure(
        figsize=FIGSIZE,
        facecolor="white"
    )

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 1],
        hspace=0.34,
        wspace=0.1,
        left=0.005,
        right=0.995,
        top=0.88,
        bottom=0.05
    )

    panels = [
        (
            fig.add_subplot(gs[0, :]),
            "ALL",
            "a) All countries",
        ),
        (
            fig.add_subplot(gs[1, 0]),
            "ETH",
            "b) Ethiopia",
        ),
        (
            fig.add_subplot(gs[1, 1]),
            "SOM",
            "c) Somalia",
        ),
        (
            fig.add_subplot(gs[1, 2]),
            "SDN",
            "d) Sudan",
        ),
    ]

    # --------------------------------------------------------
    # DRAW PERCENTAGE SCALE
    # --------------------------------------------------------

    def draw_scale(ax, x, bar_w, ticks):

        y_bottom = 0.105
        y_top = 0.842

        for tick in ticks:
            xx = x + tick * bar_w

            ax.plot(
                [xx, xx],
                [y_bottom, y_top],
                transform=ax.transAxes,
                color=GRID_COLOR,
                linewidth=0.55,
                zorder=0,
            )

            ax.plot(
                [xx, xx],
                [y_bottom, y_bottom - 0.008],
                transform=ax.transAxes,
                color="#9ca3aa",
                linewidth=0.65,
            )

            ax.text(
                xx,
                y_bottom - 0.020,
                f"{int(tick * 100)}%",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=TICK_FONT,
                color="#535b63",
            )

    # --------------------------------------------------------
    # DRAW ONE PANEL
    # --------------------------------------------------------

    def draw_matrix(ax, country):

        is_all = country == "ALL"

        label_left = (
            0.115
            if is_all
            else 0.175
        )

        right = 0.985
        gap = 0.020

        col_w = (
            right
            - label_left
            - 2 * gap
        ) / 3

        xs = [
            label_left
            + i * (col_w + gap)
            for i in range(3)
        ]

        bar_fraction = (
            BAR_FRACTION_ALL
            if is_all
            else BAR_FRACTION_COUNTRY
        )

        bar_w = col_w * bar_fraction

        # ----------------------------------------------------
        # COLUMN HEADINGS
        # ----------------------------------------------------

        for kind, x in zip(TYPES, xs):

            ax.text(
                x + bar_w / 2,
                0.956,
                kind,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=HEADING_FONT,
                fontweight="bold",
                color=COLORS[kind],
            )

        # ----------------------------------------------------
        # MODEL ROWS
        # ----------------------------------------------------

        rows = {
            "Event-based": [
                0.807,
                0.732,
                0.657,
                0.582,
                0.507,
            ],
            "Probabilistic": [
                0.346,
                0.271,
                0.196,
            ],
        }

        group_y = {
            "Event-based": 0.885,
            "Probabilistic": 0.425,
        }

        # Draw scale before bars so gridlines
        # remain in the background.
        for x in xs:
            draw_scale(
                ax,
                x,
                bar_w,
                ticks=(
                    [0, 0.25, 0.50, 0.75, 1.0]
                    if is_all
                    else [0.50, 1.0]
                )
            )

        # ----------------------------------------------------
        # GROUPS AND MODELS
        # ----------------------------------------------------

        for group in GROUPS:

            ax.text(
                0.012,
                group_y[group],
                group,
                transform=ax.transAxes,
                fontsize=GROUP_FONT,
                fontweight="bold",
                va="center",
                color="#505a65",
            )

            models = [
                model
                for family in GROUPS[group]
                for model in family
            ]

            for model, y in zip(
                models,
                rows[group]
            ):

                # Model name
                ax.text(
                    0.012,
                    y,
                    LABELS.get(model, model),
                    transform=ax.transAxes,
                    va="center",
                    fontsize=MODEL_FONT,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                )

                # --------------------------------------------
                # THREE COLUMNS
                # --------------------------------------------

                for kind, x in zip(TYPES, xs):

                    match = stats[
                        (stats.country == country)
                        & (stats.modeler == model)
                        & (stats.type == kind)
                    ]

                    # Background track to 100%
                    ax.add_patch(
                        Rectangle(
                            (
                                x,
                                y - BAR_HEIGHT / 2
                            ),
                            bar_w,
                            BAR_HEIGHT,
                            transform=ax.transAxes,
                            facecolor=BACKGROUND,
                            edgecolor="none",
                            zorder=1,
                        )
                    )

                    if match.empty:

                        label = "n/a"

                    else:

                        r = match.iloc[0]

                        median = float(
                            r["median"]
                        )

                        minimum = float(
                            r["minimum"]
                        )

                        maximum = float(
                            r["maximum"]
                        )

                        fill = float(
                            np.clip(
                                median,
                                0,
                                1
                            )
                        )

                        # Colored median bar
                        if fill > 0:

                            ax.add_patch(
                                Rectangle(
                                    (
                                        x,
                                        y - BAR_HEIGHT / 2
                                    ),
                                    bar_w * fill,
                                    BAR_HEIGHT,
                                    transform=ax.transAxes,
                                    facecolor=COLORS[kind],
                                    edgecolor="none",
                                    zorder=2,
                                )
                            )

                        # Values >100%
                        if median > 1:

                            ax.plot(
                                x + bar_w,
                                y,
                                marker="D",
                                markersize=3.3,
                                color=COLORS[kind],
                                transform=ax.transAxes,
                                clip_on=False,
                                zorder=4,
                            )

                        label = range_text(
                            minimum,
                            maximum
                        )

                    # ----------------------------------------
                    # NUMERIC LABELS: ALL COUNTRIES ONLY
                    # ----------------------------------------

                    if is_all:

                        ax.text(
                            x + bar_w + 0.007,
                            y,
                            label,
                            transform=ax.transAxes,
                            va="center",
                            ha="left",
                            fontsize=VALUE_FONT,
                            color=TEXT_COLOR,
                            zorder=5,
                        )

        # ----------------------------------------------------
        # GROUP DIVIDER
        # ----------------------------------------------------

        ax.plot(
            [0.01, 0.985],
            [0.457, 0.457],
            transform=ax.transAxes,
            color="#cbd0d5",
            linewidth=0.8,
            linestyle=":",
        )

        # Subfamily separators
        for yy in [
            0.695,
            0.620,
            0.309,
        ]:

            ax.plot(
                [0.01, 0.985],
                [yy, yy],
                transform=ax.transAxes,
                color="#edf0f2",
                linewidth=0.55,
            )

    # --------------------------------------------------------
    # FOUR PANELS
    # --------------------------------------------------------

    for ax, country, title in panels:

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Panel title above the metric headings, with dedicated spacing.
        ax.text(
            0.012, 1.075, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=PANEL_FONT,
            fontweight="bold",
            color=TEXT_COLOR,
            zorder=10,
            clip_on=False,
        )

        draw_matrix(
            ax,
            country
        )
    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    for ext in ["png", "pdf"]:

        path = (
            OUTPUT
            / f"fraction_three_column_matrix_A4.{ext}"
        )

        fig.savefig(
            path,
            dpi=300,
            facecolor="white"
        )

        print("Saved:", path)

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    refs, periods = load_observations()

    models = load_models()

    long, stats = calculate(
        models,
        refs
    )

    long.to_csv(
        OUTPUT
        / "fraction_three_column_matrix_by_specification.csv",
        index=False,
    )

    stats.to_csv(
        OUTPUT
        / "fraction_three_column_matrix_summary.csv",
        index=False,
    )

    print(
        stats.to_string(index=False)
    )

    plot(
        stats,
        periods
    )


if __name__ == "__main__":
    main()

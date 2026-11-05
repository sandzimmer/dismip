
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

WD = Path(__file__).resolve().parents[1]
INPUT_DIR = WD / "data"
IDMC_DIR = WD / "dismip/new_idmc_data"
OUTPUT_DIR = WD / "dismip/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2008
END_YEAR = 2023

YEARS = np.arange(START_YEAR, END_YEAR + 1)

COUNTRIES = {
    "ETH": "Ethiopia",
    "SOM": "Somalia",
    "SDN": "Sudan"
}

COLORS = {
    "IDMC": "#D18B6A",  # Muted terracotta
    "DFO":  "#527C91"   # Desaturated steel blue
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#B5BDC4",
    "axes.linewidth": 0.8,
    "xtick.color": "#49545C",
    "ytick.color": "#49545C",
    "axes.labelcolor": "#25313B",
    "text.color": "#25313B",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def format_millions(x, pos):
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def style_axis(ax, ylabel=None):
    ax.grid(
        axis="y",
        color="#E7EBEE",
        linewidth=0.8,
        zorder=0
    )
    ax.set_axisbelow(True)
    ax.set_xlim(START_YEAR - 0.45, END_YEAR + 0.45)
    ax.set_xticks(YEARS[::2])
    ax.tick_params(axis="both", length=0, pad=7)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlabel("Year")


def save_figure(fig, filename):
    for ext in ["pdf", "png"]:
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{ext}",
            bbox_inches="tight",
            dpi=300
        )
    plt.close(fig)


def normalize_country(value):
    value = str(value).strip().lower()

    mapping = {
        "eth": "ETH",
        "ethiopia": "ETH",
        "som": "SOM",
        "somalia": "SOM",
        "sdn": "SDN",
        "sudan": "SDN",
        "ab9": "SDN"
    }

    return mapping.get(value)


# --------------------------------------------------
# 1. LOAD NEW IDMC DATA
# --------------------------------------------------

idmc = pd.read_csv(
    IDMC_DIR / "annual_country_displacements.csv"
)

required = {
    "ISO3", "Year", "displacements_all"
}

missing = required - set(idmc.columns)
if missing:
    raise ValueError(
        f"Missing IDMC columns: {missing}"
    )

idmc["ISO3"] = idmc["ISO3"].apply(normalize_country)
idmc["Year"] = pd.to_numeric(
    idmc["Year"], errors="coerce"
)
idmc["displacements_all"] = pd.to_numeric(
    idmc["displacements_all"], errors="coerce"
).fillna(0)

idmc = idmc[
    idmc["ISO3"].isin(COUNTRIES)
    & idmc["Year"].between(START_YEAR, END_YEAR)
].copy()

idmc_annual = (
    idmc.groupby(["ISO3", "Year"], as_index=False)
    ["displacements_all"].sum()
)

# Complete country-year grid for plotting
index = pd.MultiIndex.from_product(
    [COUNTRIES.keys(), YEARS],
    names=["ISO3", "Year"]
)

idmc_annual = (
    idmc_annual.set_index(["ISO3", "Year"])
    .reindex(index, fill_value=0)
    .reset_index()
)

# --------------------------------------------------
# 2. LOAD DFO COUNTRY-LEVEL DISPLACEMENTS
# --------------------------------------------------

dfo = pd.read_csv(
    INPUT_DIR / "dfo_country_displacements.csv"
)

dfo["ISO3"] = dfo["Country"].apply(normalize_country)
dfo["Began"] = pd.to_numeric(
    dfo["Began"], errors="coerce"
)
dfo["Displaced_original"] = pd.to_numeric(
    dfo["Displaced_original"], errors="coerce"
).fillna(0)

dfo = dfo[
    dfo["ISO3"].isin(COUNTRIES)
    & dfo["Began"].between(START_YEAR, END_YEAR)
].copy()

dfo_annual = (
    dfo.groupby(["ISO3", "Began"], as_index=False)
    ["Displaced_original"].sum()
    .rename(columns={"Began": "Year"})
)

dfo_annual = (
    dfo_annual.set_index(["ISO3", "Year"])
    .reindex(index, fill_value=0)
    .reset_index()
)

# --------------------------------------------------
# 3. COMBINE ANNUAL DISPLACEMENTS
# --------------------------------------------------

annual = idmc_annual.merge(
    dfo_annual,
    on=["ISO3", "Year"],
    how="outer"
).fillna(0)

annual = annual.rename(columns={
    "displacements_all": "IDMC",
    "Displaced_original": "DFO"
})

annual["Country"] = annual["ISO3"].map(COUNTRIES)

annual.to_csv(
    OUTPUT_DIR / "annual_displacements_idmc_dfo_2008_2023.csv",
    index=False
)

all_annual = (
    annual.groupby("Year", as_index=False)
    [["IDMC", "DFO"]].sum()
)

# --------------------------------------------------
# 4. FIGURE 1: TOTAL ANNUAL DISPLACEMENTS
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

for source in ["DFO", "IDMC"]:
    ax.plot(
        all_annual["Year"],
        all_annual[source],
        color=COLORS[source],
        linewidth=2.5,
        marker="o",
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.9,
        label=source,
        zorder=3
    )

style_axis(ax, "Reported displacements")
ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
ax.set_ylim(bottom=0)

ax.set_title(
    "Annual flood-induced displacements",
    loc="left",
    pad=18
)

ax.text(
    0, 1.02,
    "Ethiopia, Somalia and Sudan combined · 2008–2023",
    transform=ax.transAxes,
    fontsize=10,
    color="#64717C"
)

ax.legend(
    frameon=False,
    loc="upper right",
    ncol=2,
    fontsize=11
)

fig.tight_layout()
save_figure(
    fig,
    "displacements_comp_idmc_dfo_2008_2023"
)

# --------------------------------------------------
# 5. FIGURE 2: COUNTRY-SPECIFIC DISPLACEMENTS
# --------------------------------------------------

fig, axes = plt.subplots(
    3, 1,
    figsize=(6, 5.5),
    sharex=True
)

for ax, (iso, country) in zip(axes, COUNTRIES.items()):

    subset = annual[annual["ISO3"] == iso]

    for source in ["DFO", "IDMC"]:
        ax.plot(
            subset["Year"],
            subset[source],
            color=COLORS[source],
            linewidth=2.1,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=source,
            zorder=3
        )

    style_axis(ax, "Displacements")
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_millions)
    )
    ax.set_ylim(bottom=0)
    ax.set_title(
        country,
        loc="left",
        fontsize=12,
        pad=9
    )

axes[0].set_title(
    "Ethiopia",
    loc="left",
    fontsize=12,
    pad=9
)

fig.suptitle(
    "Annual flood-induced displacements by country",
    fontsize=16,
    fontweight="semibold",
    x=0.08,
    ha="left",
    y=0.995
)

handles = [
    Line2D(
        [0], [0],
        color=COLORS[source],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=source
    )
    for source in ["IDMC", "DFO"]
]

fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.94, 1.0),
    frameon=False,
    ncol=2
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

save_figure(
    fig,
    "displacements_by_country_idmc_dfo_2008_2023"
)

# --------------------------------------------------
# 6. FIGURE 3: ANNUAL REPORTED EVENT COUNTS
# --------------------------------------------------

# Requires original classified GIDD records,
# since annual_country_displacements.csv contains
# aggregated totals rather than unique event IDs.

records_path = IDMC_DIR / "classified_records.csv"

if records_path.exists():

    records = pd.read_csv(records_path)

    records["ISO3"] = records["ISO3"].apply(
        normalize_country
    )
    records["Year"] = pd.to_numeric(
        records["Year"], errors="coerce"
    )

    records = records[
        records["ISO3"].isin(COUNTRIES)
        & records["Year"].between(START_YEAR, END_YEAR)
    ].copy()

    # Use a country-event key to avoid collisions
    # between different countries.
    idmc_counts = (
        records.dropna(subset=["Event ID"])
        .groupby("Year")
        .apply(
            lambda x: x[
                ["ISO3", "Event ID"]
            ].drop_duplicates().shape[0]
        )
        .reindex(YEARS, fill_value=0)
    )

    # Assumes one DFO row represents one
    # country-specific flood event.
    dfo_counts = (
        dfo.groupby("Began")
        .size()
        .reindex(YEARS, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    width = 0.36

    ax.bar(
        YEARS - width / 2,
        idmc_counts.values,
        width=width,
        color=COLORS["IDMC"],
        label="IDMC",
        zorder=3
    )

    ax.bar(
        YEARS + width / 2,
        dfo_counts.values,
        width=width,
        color=COLORS["DFO"],
        label="DFO",
        zorder=3
    )

    style_axis(ax, "Number of reported events")
    ax.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    ax.set_title(
        "Annual reported flood-displacement events",
        loc="left",
        pad=18
    )

    ax.text(
        0, 1.02,
        "Ethiopia, Somalia and Sudan · 2008–2023",
        transform=ax.transAxes,
        fontsize=10,
        color="#64717C"
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "reported_events_comp_idmc_dfo_2008_2023"
    )

else:
    print(
        "Skipping event-count figure: "
        "classified_records.csv not found."
    )

# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\nTotal reported displacements (2008–2023):")

print(
    annual.groupby("Country")[["IDMC", "DFO"]]
    .sum()
    .round(0)
    .to_string()
)

print("\nAnnual average displacements:")

print(
    (
        annual.groupby("Country")[["IDMC", "DFO"]]
        .sum() / len(YEARS)
    )
    .round(0)
    .to_string()
)

print("\nFigures saved to:", OUTPUT_DIR)


# --------------------------------------------------
# 8. NUMBER OF REPORTED EVENTS (2008–2023)
# --------------------------------------------------

# DFO: number of rows per country
dfo_event_counts = (
    dfo.groupby("ISO3")
    .size()
    .reindex(COUNTRIES.keys(), fill_value=0)
)

# IDMC: number of unique Event IDs per country
idmc_event_counts = (
    records.groupby("ISO3")["Event ID"]
    .nunique()
    .reindex(COUNTRIES.keys(), fill_value=0)
)

event_summary = pd.DataFrame({
    "Country": list(COUNTRIES.values()),
    "DFO_events": dfo_event_counts.values,
    "IDMC_events": idmc_event_counts.values
})

# Combined total across all three countries
event_summary.loc[len(event_summary)] = {
    "Country": "ALL",
    "DFO_events": len(dfo),
    "IDMC_events": records["Event ID"].nunique()
}

print("\nNumber of reported events (2008–2023):")
print(event_summary.to_string(index=False))

event_summary.to_csv(
    OUTPUT_DIR / "event_counts_idmc_dfo_2008_2023.csv",
    index=False
)




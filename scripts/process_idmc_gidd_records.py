
from pathlib import Path
from itertools import combinations
import pandas as pd
import re

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "dismip/new_idmc_data"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

START_YEAR = 2008
END_YEAR = 2025

# --------------------------------------------------
# 1. LOAD ALL GIDD FILES
# --------------------------------------------------
files = list(DATA_DIR.glob("*.xlsx"))

dfs = [
    pd.read_excel(file, sheet_name="1_Disaggregated_Data")
    for file in files
]

df = pd.concat(dfs, ignore_index=True)

# Remove duplicate GIDD records across files
df = df.drop_duplicates(subset="ID")

df = df[
    df["Year"].between(START_YEAR, END_YEAR)
    & df["Hazard type"].astype(str).str.contains("Flood", case=False, na=False)
].copy()

print("Records:", len(df))
print("Unique events:", df["Event ID"].nunique())

# --------------------------------------------------
# 2. CLASSIFY POSSIBLE FLOOD TYPES
# --------------------------------------------------
def classify_flood(row):
    text = " ".join(
        str(row[col])
        for col in ["Event name", "Hazard sub type", "Other event sub type"]
        if pd.notna(row[col])
    ).lower()

    flash = bool(re.search(r"flash flood|flashflood", text))
    pluvial = bool(re.search(
        r"pluvial|urban flood|surface.water flood|rainfall flooding",
        text
    ))
    landslide = bool(re.search(
        r"landslide|mudslide|mudflow|debris flow|mass movement",
        text
    ))

    types = []
    if flash:
        types.append("Flash")
    if pluvial:
        types.append("Pluvial")
    if landslide:
        types.append("Landslide / mass movement")

    return ", ".join(types) if types else "Not explicitly identified"


df["flood_class"] = df.apply(classify_flood, axis=1)

# One row per Event ID
events = df.groupby("Event ID").agg(
    year=("Year", "first"),
    country=("Country", "first"),
    event_name=("Event name", "first"),
    flood_types=("flood_class", lambda x: "; ".join(
        sorted(set(x) - {"Not explicitly identified"})
    ) or "Not explicitly identified")
).reset_index()

print(events[events['flood_types']== 'Flash'])

events["mixed_event"] = events["flood_types"].str.contains(";")

print("\nFlood classifications (unique events):")
print(events["flood_types"].value_counts())
print("Potential mixed events:", events["mixed_event"].sum())

# --------------------------------------------------
# 3. COMPARE LOCATION NAMES
# --------------------------------------------------

import re
import pandas as pd

def split_locations(value):
    """Split location names and normalize them for comparison."""
    if pd.isna(value):
        return []

    names = re.split(r"[,;]", str(value))

    return [
        re.sub(r"\s+", " ", name.strip().lower())
        for name in names
        if name.strip()
    ]

df["location"] = df["Locations name"].apply(split_locations)

# Event start date
df["event_start"] = pd.to_datetime(
    df["Event start date"], errors="coerce", dayfirst=True
)

# Use calendar year from Event start date
df["Year"] = df["event_start"].dt.year

df = df[df["Year"].between(2008, 2025)].copy()

df["location"] = df["Locations name"].apply(split_locations)

locations = (
    df.explode("location")
    .dropna(subset=["location", "Event ID"])
    [["ISO3", "Country", "Year", "location", "Event ID", "event_start"]]
    .drop_duplicates()
)

# One start date per Event ID and location
locations = (
    locations.groupby(
        ["ISO3", "Country", "Year", "location", "Event ID"],
        as_index=False
    )
    .agg(event_start=("event_start", "min"))
)


def count_separate_events(group):
    dates = (
        group["event_start"]
        .dropna()
        .sort_values()
        .tolist()
    )

    separate_dates = []

    for date in dates:
        if not separate_dates or (date - separate_dates[-1]).days >= 60:
            separate_dates.append(date)

    return pd.Series({
        "n_event_ids": group["Event ID"].nunique(),
        "n_separate_events": len(separate_dates),
        "n_missing_dates": group["event_start"].isna().sum()
    })


location_counts = (
    locations.groupby(
        ["ISO3", "Country", "Year", "location"]
    )
    .apply(count_separate_events, include_groups=False)
    .reset_index()
)

# Classification
location_counts["category"] = "One event"

location_counts.loc[
    location_counts["n_separate_events"] >= 2,
    "category"
] = "Multiple events at least 30 days apart"

location_counts.loc[
    (location_counts["n_event_ids"] > 1)
    & (location_counts["n_separate_events"] < 2),
    "category"
] = "Multiple Event IDs within 30 days"

# Summary
print("\nLocation-event summary (2008–2025):")
print(location_counts["category"].value_counts())

# Country-year summary
country_year = (
    location_counts.groupby(["ISO3", "Country", "Year"])
    .agg(
        total_locations=("location", "nunique"),
        locations_one_event=(
            "category", lambda x: (x == "One event").sum()
        ),
        locations_multiple_events_30d=(
            "category",
            lambda x: (x == "Multiple events at least 30 days apart").sum()
        )
    )
    .reset_index()
)

location_counts.to_csv(
    OUTPUT_DIR / "location_event_counts.csv", index=False
)

country_year.to_csv(
    OUTPUT_DIR / "country_year_summary.csv", index=False
)
# -------------------------------------------
# 4. PREVENTIVE EVACUATIONS
# --------------------------------------------------
print("\nDisplacement occurred values:")
print(df["Displacement occurred"].value_counts(dropna=False))

status = df["Displacement occurred"].astype("string").str.lower()

df["preventive"] = status.str.contains(
    "without pre.?emptive|without preventive", regex=True, na=False
)

df["evacuation_info_available"] = df["Displacement occurred"].notna()

print("\nRecords with information:",
      df["evacuation_info_available"].sum())
print("Records without information:",
      (~df["evacuation_info_available"]).sum())
print("Records mentioning preventive evacuation:",
      df["preventive"].sum())



# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------
events.to_csv(OUTPUT_DIR / "event_classification.csv", index=False)
#overlaps.to_csv(OUTPUT_DIR / "same_location_multiple_events.csv", index=False)
country_year.to_csv(OUTPUT_DIR / "country_year_summary.csv", index=False)
df.to_csv(OUTPUT_DIR / "classified_records.csv", index=False)

print("\nFinished. Results saved to:", OUTPUT_DIR)



# --------------------------------------------------
# 5. ANNUAL COUNTRY-LEVEL DISPLACEMENT TOTALS
# --------------------------------------------------

# Check the displacement column name in your GIDD files
DISPLACEMENT_COL = "Total figures"

# Convert displacement values to numeric
df[DISPLACEMENT_COL] = pd.to_numeric(
    df[DISPLACEMENT_COL], errors="coerce"
)

# Identify records associated with excluded flood types
# Uses the flood_class column created in Section 2
excluded_types = ["Flash", "Pluvial", "Landslide / mass movement"]

df["excluded_flood_type"] = df["flood_class"].apply(
    lambda x: any(t in str(x) for t in excluded_types)
)

# Retain records without explicitly identified excluded flood types
df["included_flood_type"] = ~df["excluded_flood_type"]

# Create displacement categories
df["displacements_all"] = df[DISPLACEMENT_COL]

df["displacements_filtered"] = df[DISPLACEMENT_COL].where(
    df["included_flood_type"], 0
)

df["displacements_preventive"] = df[DISPLACEMENT_COL].where(
    df["preventive"], 0
)

df["displacements_filtered_preventive"] = df[DISPLACEMENT_COL].where(
    df["included_flood_type"] & df["preventive"], 0
)

# Aggregate by country and year
annual_displacements = (
    df.groupby(["ISO3", "Country", "Year"], as_index=False)
    .agg(
        displacements_all=("displacements_all", "sum"),
        displacements_filtered=("displacements_filtered", "sum"),
        displacements_preventive=("displacements_preventive", "sum"),
        displacements_filtered_preventive=(
            "displacements_filtered_preventive", "sum"
        ),
        n_records=("ID", "nunique"),
        n_events=("Event ID", "nunique")
    )
)

# Sort results
annual_displacements = annual_displacements.sort_values(
    ["ISO3", "Year"]
)

# Save results
annual_displacements.to_csv(
    OUTPUT_DIR / "annual_country_displacements.csv",
    index=False
)

print("\nAnnual country-level displacement totals:")
print(annual_displacements.head(20))




from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "dismip/new_idmc_data"

df = pd.read_csv(
    DATA_DIR / "annual_country_displacements.csv"
)

COUNTRIES = ["ETH", "SOM", "SDN"]
PERIODS = [(2009, 2021), (2008, 2025)]

# Include all displacement categories created previously
displacement_cols = [
    "displacements_all",
    "displacements_filtered"
]

results = []

for start, end in PERIODS:
    n_years = end - start + 1

    subset = df[
        df["ISO3"].isin(COUNTRIES)
        & df["Year"].between(start, end)
    ].copy()

    # Sum displacements by country
    country_totals = (
        subset.groupby("ISO3")[displacement_cols]
        .sum()
        .reindex(COUNTRIES, fill_value=0)
    )

    # Add combined total for all three countries
    country_totals.loc["ALL"] = country_totals.sum()

    # Divide by total number of calendar years
    aad = country_totals / n_years

    for country, values in aad.iterrows():
        results.append({
            "period": f"{start}-{end}",
            "country": country,
            "n_years": n_years,
            **values.to_dict()
        })

aad_overview = pd.DataFrame(results)

aad_overview.to_csv(
    DATA_DIR / "aad_overview_periods.csv",
    index=False
)

print(aad_overview.to_string(index=False))




# ============================================================
# 6. PIE CHARTS – FLOOD TYPES AND REPEATED EVENTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

THRESHOLD_DAYS = 60

COLORS = {
    "Flash": "#4baa9a",
    "Pluvial": "#4b8aaa",
    "Landslide / mass movement": "#da9e30",
    "Mixed events": "#af6508",
    "Not explicitly identified": "#bdbdbd",
    "One event": "#4baa9a",
    "Multiple Event IDs within 30 days": "#da9e30",
    "Multiple events at least 30 days apart": "#4b5baa",
    "Same Event ID within 30 days": "#4baa9a",
    "Same Event ID at least 30 days apart": "#af6508",
    "Same Event ID, one start date": "#bdbdbd",
    "Missing start date": "#8d8d8d",
}



def plot_pie(counts, title, filename):
    """Create a pie chart with non-overlapping labels."""

    counts = counts[counts > 0]

    if counts.empty:
        print(f"No data for {title}")
        return

    labels = counts.index.tolist()
    values = counts.to_numpy()
    total = values.sum()

    colors = [
        COLORS.get(label, "#999999")
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Only show percentages for sufficiently large slices
    def autopct_format(pct):
        if pct < 6:
            return ""
        return f"{pct:.1f}%"

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=autopct_format,
        pctdistance=0.68,
        radius=0.95,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.5
        },
        textprops={
            "fontsize": 11,
            "fontweight": "bold"
        }
    )

    # Improve readability of percentage labels
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # All categories, counts and percentages in legend
    legend_labels = [
        f"{label}: {value:,} ({value / total * 100:.1f}%)"
        for label, value in zip(labels, values)
    ]

    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=10,
        labelspacing=1.2
    )

    ax.set_title(
        f"{title}\nTotal: {total:,}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.axis("equal")

    # Save as PNG and PDF
    for extension in ["png", "pdf"]:
        fig.savefig(
            OUTPUT_DIR / f"{Path(filename).stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25
        )

    plt.close(fig)

    plt.close(fig)



# ============================================================
# PIE CHART 1 – FLOOD TYPES
# ============================================================

def classify_event_type(value):
    """Assign each unique Event ID to one exclusive category."""

    types = {
        t.strip()
        for t in str(value).split(";")
        if t.strip()
        and t.strip() != "Not explicitly identified"
    }

    if len(types) > 1:
        return "Mixed events"

    if not types:
        return "Not explicitly identified"

    return next(iter(types))


flood_events = events.copy()

flood_events["pie_category"] = (
    flood_events["flood_types"].apply(classify_event_type)
)

flood_counts = (
    flood_events["pie_category"]
    .value_counts()
    .reindex(
        [
            "Flash",
            "Pluvial",
            "Landslide / mass movement",
            "Mixed events",
            "Not explicitly identified"
        ],
        fill_value=0
    )
)

print("\nFlood-type classification:")
print(flood_counts)

plot_pie(
    flood_counts,
    "Flood-type classification (unique Event IDs)",
    "pie_flood_types.png"
)


# ============================================================
# PIE CHART 2 – MULTIPLE EVENT IDS AT SAME LOCATION
# ============================================================

# Recalculate using the actual 30-day threshold.
# The original script uses 60 days in count_separate_events().

def classify_location_events(group):

    dates = (
        group["event_start"]
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )

    n_ids = group["Event ID"].nunique()

    if n_ids <= 1:
        return "One event"

    if len(dates) < 2:
        return "Missing start date"

    # Use consecutive start dates to identify separate events.
    n_separate = 1

    for previous, current in zip(dates[:-1], dates[1:]):
        if (current - previous).days >= THRESHOLD_DAYS:
            n_separate += 1

    if n_separate >= 2:
        return "Multiple events at least 30 days apart"

    return "Multiple Event IDs within 30 days"


location_categories = (
    locations.groupby(
        ["ISO3", "Country", "Year", "location"]
    )
    .apply(classify_location_events, include_groups=False)
    .rename("category")
    .reset_index()
)

location_counts_pie = (
    location_categories["category"]
    .value_counts()
    .reindex(
        [
            "One event",
            "Multiple Event IDs within 30 days",
            "Multiple events at least 30 days apart",
            "Missing start date"
        ],
        fill_value=0
    )
)

print("\nSame-location event classification:")
print(location_counts_pie)

plot_pie(
    location_counts_pie,
    "Repeated flood events at the same location",
    "pie_location_events.png"
)


# ============================================================
# PIE CHART 3 – SAME EVENT ID AND START-DATE DIFFERENCES
# ============================================================

# One classification per unique Event ID.
# This checks whether records associated with the same ID
# have different start dates.

# --------------------------------------------------
files = list(DATA_DIR.glob("*.xlsx"))

dfs = [
    pd.read_excel(file, sheet_name="1_Disaggregated_Data")
    for file in files
]

df = pd.concat(dfs, ignore_index=True)

# Remove duplicate GIDD records across files
df = df.drop_duplicates(subset="ID")

df = df[
    df["Year"].between(START_YEAR, END_YEAR)
    & df["Hazard type"].astype(str).str.contains("Flood", case=False, na=False)
].copy()

print(df.columns)

event_dates = (
    df.dropna(subset=["Event ID"])
    .groupby("Event ID")["Event start date"]
    .agg(
        lambda x: sorted(
            pd.to_datetime(x.dropna()).unique()
        )
    )
)


def classify_same_event_id(dates):

    if len(dates) == 0:
        return "Missing start date"

    if len(dates) == 1:
        return "Same Event ID, one start date"

    difference = (
        pd.Timestamp(dates[-1])
        - pd.Timestamp(dates[0])
    ).days

    if difference < THRESHOLD_DAYS:
        return "Same Event ID within 30 days"

    return "Same Event ID at least 30 days apart"


same_id_categories = event_dates.apply(
    classify_same_event_id
)

same_id_counts = (
    same_id_categories
    .value_counts()
    .reindex(
        [
            "Same Event ID, one start date",
            "Same Event ID within 30 days",
            "Same Event ID at least 30 days apart",
            "Missing start date"
        ],
        fill_value=0
    )
)

print("\nSame Event ID classification:")
print(same_id_counts)

plot_pie(
    same_id_counts,
    "Start-date differences within the same Event ID",
    "pie_same_event_id.png"
)

print("\nAll three pie charts saved to:", OUTPUT_DIR)





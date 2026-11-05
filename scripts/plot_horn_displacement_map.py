#!/usr/bin/env python3
"""Horn of Africa map with fully in-frame displacement charts (2008–2021).

GADM and observation paths are inherited from the original plotting script.
All time series use log10(displacements + 1) on the SAME scale, including
conflict-related IDPs on the right axis. No observations are invented.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

WD = Path(__file__).resolve().parents[1]
INPUT_DIR = WD / 'dismip/dismip_data'
OUTPUT_DIR = WD / 'dismip/output'
GADM_FILE = Path(str(Path(__file__).resolve().parents[1] / 'data/geospatial/gadm36_levels.gpkg'))
IDMC_FILE = WD / 'dismip/new_idmc_data/annual_country_displacements.csv'
DFO_FILE = WD / 'data/dfo_country_displacements.csv'
IDPS_FILE = INPUT_DIR / 'idps.xlsx'
YEARS = np.arange(2008, 2022)
COUNTRIES = {'ETH': 'Ethiopia', 'SDN': 'Sudan', 'SOM': 'Somalia'}
COLORS = {'ETH': '#da9e30', 'SDN': '#4baa9a', 'SOM': '#66023c'}

# Figure coordinates: charts are inside the geographic map, not separate panels.
# Charts may overlap country outlines; their near-white backgrounds preserve readability.
MAP_EXTENT = (17, 57, -5.5, 24)  # west, east, south, north (degrees)
MAP_POS = [0.025, 0.115, 0.95, 0.85]
AFRICA_POS = [0.017, 0.75, 0.17, 0.17]
CHART_POS = {
    'ETH': [0.68, 0.750, 0.275, 0.185],  # upper left, beside Africa inset
    'SDN': [0.065, 0.185, 0.275, 0.185],  # lower left
    'SOM': [0.680, 0.185, 0.275, 0.185],  # lower right
}


def read_observations():
    obs = pd.read_csv(IDMC_FILE)
    obs['year'] = pd.to_numeric(obs['Year'], errors='coerce')
    obs['displacements'] = pd.to_numeric(obs['displacements_filtered'], errors='coerce')
    # Updated IDMC CSV uses ISO3 codes; Country may contain full names.
    if 'ISO3' not in obs.columns:
        raise ValueError(f'Updated IDMC file missing ISO3 column: {IDMC_FILE}')
    obs['country'] = (obs['ISO3'].astype(str).str.strip().str.upper()
                      .replace({'AB9': 'SDN'}))
    obs = obs[obs['country'].isin(COUNTRIES) & obs['year'].isin(YEARS)].copy()
    if obs.empty:
        raise ValueError('No updated IDMC observations matched ETH/SOM/SDN and 2008–2021.')
    idmc = obs.groupby(['country', 'year'])['displacements'].sum(min_count=1).unstack(0)
    print('Updated IDMC displacement totals by country (2008–2021):')
    print(idmc.sum(axis=0, min_count=1).to_string())
    for iso in COUNTRIES:
        if iso not in idmc.columns or idmc[iso].notna().sum() == 0:
            raise ValueError(f'Updated IDMC data missing for {iso}; check ISO3 and Year columns.')

    dfo = pd.read_csv(DFO_FILE)
    dfo['country'] = dfo['Country'].replace({v: k for k, v in COUNTRIES.items()})
    dfo['year'] = pd.to_numeric(dfo['Began'], errors='coerce')
    dfo['Displaced'] = pd.to_numeric(dfo['Displaced_original'], errors='coerce')
    dfo = dfo[dfo['country'].isin(COUNTRIES)]
    dfo = dfo.groupby(['country', 'year'])['Displaced'].sum().unstack(0)

    idps = pd.read_excel(IDPS_FILE)
    required = {'ISO3', 'year', 'IDPs'}
    if not required.issubset(idps.columns):
        raise ValueError(f'Missing IDP columns: {required - set(idps.columns)}')
    idps['ISO3'] = idps['ISO3'].astype(str).str.upper()
    idps['year'] = pd.to_numeric(idps['year'], errors='coerce')
    idps['IDPs'] = pd.to_numeric(idps['IDPs'], errors='coerce')
    idps = idps.groupby(['ISO3', 'year'])['IDPs'].mean().unstack(0)

    def yearly(frame):
        return frame.reindex(index=YEARS, columns=list(COUNTRIES))

    return yearly(idmc), yearly(dfo), yearly(idps)


def read_gadm():
    world = gpd.read_file(GADM_FILE, layer='level0')
    col = next((c for c in world.columns if c.upper() == 'GID_0'), None)
    if col is None:
        raise KeyError('GADM level0 missing GID_0')
    world['_iso3'] = world[col].astype(str).str.upper()
    if world.crs is None:
        raise ValueError('GADM level0 layer has no CRS')
    world = world.to_crs('EPSG:4326')
    missing = set(COUNTRIES) - set(world['_iso3'])
    if missing:
        raise ValueError(f'Study countries missing from GADM: {missing}')
    return world


def logplus1(values):
    """Missing observations remain NaN; zero displacement is plotted at zero."""
    values = np.asarray(values, dtype=float)
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError('Displacements must be nonnegative for log10(x+1).')
    return np.log10(values + 1)


def plot():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idmc, dfo, idps = read_observations()
    world = read_gadm()
    selected = world[world['_iso3'].isin(COUNTRIES)]

    # One identical vertical scale for flood bars and conflict IDP lines.
    all_values = np.concatenate([frame.to_numpy().ravel() for frame in (idmc, dfo, idps)])
    finite = all_values[np.isfinite(all_values)]
    upper = max(1.0, np.ceil(np.log10(max(finite.max(), 1) + 1))) if finite.size else 1.0
    tick_exponents = np.arange(0, int(upper) + 1)
    tick_labels = ['0' if i == 0 else f'$10^{i}$' for i in tick_exponents]

    fig = plt.figure(figsize=(12.5, 9.0), facecolor='white')
    ax = fig.add_axes(MAP_POS, zorder=1)
    west, east, south, north = MAP_EXTENT
    region = world.cx[west:east, south:north]
    region.plot(ax=ax, facecolor='#f1f1ee', edgecolor='#aeb4b2',
                linewidth=0.5, zorder=1)
    for iso, color in COLORS.items():
        selected[selected['_iso3'].eq(iso)].plot(
            ax=ax, facecolor=color, edgecolor='#30393c', linewidth=1.0,
            alpha=0.72, zorder=3)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    # Match figure and geographic aspect so Matplotlib does not shrink the map frame.
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    # Thin frame surrounds the full main map, not only the study countries.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#454e50')
        spine.set_linewidth(1.05)
    for iso, x0, y0 in [('SDN', 30.4, 17.5), ('ETH', 39.2, 9.1), ('SOM', 47.5, 6.2)]:
        ax.text(x0, y0, COUNTRIES[iso], fontsize=11, fontweight='bold',
                ha='center', va='center', color='#20272a', zorder=10)

    # GADM Africa countries only; avoid Europe/Arabia in the locator.
    africa_codes = set('DZA AGO BEN BWA BFA BDI CMR CPV CAF TCD COM COD COG CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE ESH'.split())
    africa = world[world['_iso3'].isin(africa_codes)]
    loc = fig.add_axes(AFRICA_POS, facecolor='white', zorder=15)
    africa.plot(ax=loc, facecolor='#dfe3df', edgecolor='white', linewidth=0.2)
    loc.add_patch(Rectangle((west, south), east-west, north-south,
                            fill=False, edgecolor='#a43c38', linewidth=1.5,
                            zorder=20))
    loc.set_xlim(-20, 63)
    loc.set_ylim(-36, 39)
    loc.set_aspect('equal')
    loc.set_xticks([])
    loc.set_yticks([])
    for spine in loc.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color('#51595a')

    x = np.arange(len(YEARS))
    for iso, position in CHART_POS.items():
        chart = fig.add_axes(position, zorder=16)
        chart.set_facecolor((1, 1, 1, 0.92))  # Mostly opaque: readable over country outlines
        a = logplus1(idmc[iso].to_numpy(dtype=float))
        b = logplus1(dfo[iso].to_numpy(dtype=float))
        c = logplus1(idps[iso].to_numpy(dtype=float))

        # Log+1 is transformed BEFORE plotting: bars have a valid zero baseline.
        # Missing values are omitted, not silently replaced with zero.
        valid_a = np.isfinite(a)
        valid_b = np.isfinite(b)
        chart.bar(x[valid_a] + 0.19, a[valid_a], width=0.37,
                  facecolor=COLORS[iso], edgecolor=COLORS[iso],
                  linewidth=0.55, zorder=3)
        chart.bar(x[valid_b] - 0.19, b[valid_b], width=0.37,
                  facecolor='white', edgecolor=COLORS[iso], hatch='////',
                  linewidth=0.7, zorder=3)
        chart.set_xlim(-0.7, len(YEARS) - 0.3)
        chart.set_ylim(0, upper)
        chart.yaxis.set_major_locator(FixedLocator(tick_exponents))
        chart.yaxis.set_major_formatter(FixedFormatter(tick_labels))
        chart.set_xticks([0, 4, 8, 12])
        chart.set_xticklabels([2008, 2012, 2016, 2020])
        chart.tick_params(axis='both', labelsize=9, length=2.5, pad=2)
        chart.grid(axis='y', alpha=0.20, linewidth=0.55, zorder=0)
        chart.spines[['top', 'right']].set_visible(False)
        chart.spines[['left', 'bottom']].set_color('#7a7f80')
        chart.set_title(COUNTRIES[iso], loc='center', fontsize=14,
                        fontweight='bold', pad=3)

        # Conflict IDPs use the very SAME y-axis as flood displacement bars.
        chart.plot(x, c, color='#20272a', linestyle='--', marker='o',
                   markersize=2.4, linewidth=1.35, zorder=6)
        chart.set_ylabel(r'log$_{10}$(people)', fontsize=9, labelpad=2)

    # One shared legend, inside the bottom strip of the main geographic frame.
    legend_handles = [
        Patch(facecolor='#8b9696', edgecolor='#566060', label='IDMC flood displacements'),
        Patch(facecolor='white', edgecolor='#566060', hatch='////', label='DFO flood displacements'),
        Line2D([0], [0], color='#20272a', linestyle='--', marker='o',
               markersize=4, linewidth=1.5, label='Conflict-induced IDPs (IDMC)'),
    ]
    fig.legend(handles=legend_handles, loc='center',
               bbox_to_anchor=(0.50, 0.135), ncol=3, frameon=True,
               facecolor='white', edgecolor='#9fa7a7', fontsize=11,
               handlelength=2.5, columnspacing=2.2)
    for ext in ('pdf', 'png'):
        path = OUTPUT_DIR / f'horn_of_africa_displacement_map_framed.{ext}'
        fig.savefig(path, dpi=350, facecolor='white', bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot()

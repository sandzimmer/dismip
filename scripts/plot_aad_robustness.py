
#!/usr/bin/env python3
"""Four-panel AAD comparison with grouped model families.

Top: all countries.
Bottom: Ethiopia, Somalia, Sudan.

Event-based models use 2008–2021 observational AAD.
Probabilistic models use long-term observational AAD.

DFO displacement estimates are averaged across three allocation methods.
The long-term DFO denominator includes all calendar years in the
historical period, including years without recorded events.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

AAD_FILE = ROOT / 'dismip/dismip_data/aad.csv'
DFO_FILE = ROOT / 'data/dfo_country_displacements.csv'
IDMC_FILE = ROOT / 'dismip/new_idmc_data/annual_country_displacements.csv'

OUT = ROOT / 'dismip/output'

STUDY_START = 2008
STUDY_END = 2021

# Specify archive boundaries if known.
# Otherwise, the first and last recorded event years are used.
DFO_FULL_START = None
DFO_FULL_END = None

REQUIRE_PRE_STUDY_DFO = True


# ============================================================
# MODEL GROUPS AND COLORS
# ============================================================

GROUPS = {
    'Event-based': [
        ('KAM', 'ZIM-1M'),
        ('ZIM-D',),
        ('RIE-0', 'RIE-FLO')
    ],
    'Probabilistic': [
        ('MEI-CAPRA', 'MEI-IVM'),
        ('CIMA',)
    ]
}

NAMES = {
    'MEI-CAPRA': 'MEI-a',
    'MEI-IVM': 'MEI-b'
}

FAMILY_COLORS = {
    'KAM': '#6d9e94',
    'ZIM-1M': '#6d9e94',
    'ZIM-D': '#d29b5d',
    'RIE-0': '#7393be',
    'RIE-FLO': '#7393be',
    'MEI-CAPRA': '#c48d62',
    'MEI-IVM': '#c48d62',
    'CIMA': '#9480ae'
}

CALIBRATED = {
    'ZIM-D',
    'RIE-0',
    'RIE-FLO'
}

IDMC_COLOR = '#853d59'
DFO_COLOR = '#3c7186'
IDMC_EXCL_COLOR = '#b87591'
IDMC_RECENT_COLOR = '#653047'
RECENT_START = 2017
RECENT_END = 2025
EXCLUDED_YEAR = 2020

# Bar positioning
FAMILY_STEP = 0.62
FAMILY_GAP = 0.01
BAR_WIDTH = 0.46

# Figure layout
FIG_WIDTH = 13
FIG_HEIGHT = 9

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# ============================================================
# LOAD DFO DATA
# ============================================================

def read_dfo():

    df = pd.read_csv(DFO_FILE)

    cols = [
        'Displaced_area',
        'Displaced_population',
        'Displaced_exposure'
    ]

    if all(c in df.columns for c in cols):

        df['Displaced'] = (
            df[cols]
            .apply(pd.to_numeric, errors='coerce')
            .mean(axis=1)
        )

    elif 'Displaced' in df.columns:

        df['Displaced'] = pd.to_numeric(
            df['Displaced'],
            errors='coerce'
        )

    else:
        raise ValueError(
            'DFO CSV needs the three Displaced_* columns '
            'or a Displaced column.'
        )

    country_col = next(
        (
            c for c in ('country', 'Country', 'ISO3')
            if c in df.columns
        ),
        None
    )

    if country_col is None:
        raise ValueError('Missing country column.')

    df['country'] = df[country_col].replace({
        'Ethiopia': 'ETH',
        'Somalia': 'SOM',
        'Sudan': 'SDN'
    })

    df = df[
        df['country'].isin(['ETH', 'SOM', 'SDN'])
    ].copy()

    if 'Began' not in df.columns:
        raise ValueError('Missing Began column.')

    numeric = pd.to_numeric(
        df['Began'],
        errors='coerce'
    )

    dates = pd.to_datetime(
        df['Began'],
        errors='coerce'
    )

    df['year'] = np.where(
        numeric.between(1800, 2100),
        numeric,
        dates.dt.year
    )

    df['year'] = pd.to_numeric(
        df['year'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['year', 'Displaced']
    )

    df['year'] = df['year'].astype(int)

    observed_min = int(df['year'].min())
    observed_max = int(df['year'].max())

    first = (
        DFO_FULL_START
        if DFO_FULL_START is not None
        else observed_min
    )

    last = (
        DFO_FULL_END
        if DFO_FULL_END is not None
        else observed_max
    )

    if first > observed_min or last < observed_max:
        raise ValueError(
            'Configured DFO period excludes observed events.'
        )

    n_years = last - first + 1

    print('\n--- DFO ---')
    print(
        f'Available event years: '
        f'{observed_min}–{observed_max}'
    )
    print(
        f'Long-term averaging period: '
        f'{first}–{last} ({n_years} years)'
    )
    print(
        f'Total displacements: '
        f'{df["Displaced"].sum():,.0f}'
    )

    if REQUIRE_PRE_STUDY_DFO and first >= STUDY_START:
        raise ValueError(
            'DFO data contain no pre-2008 records. '
            'A historical AAD cannot be calculated '
            'from study-period data alone.'
        )

    for country in ['ETH', 'SOM', 'SDN']:

        subset = df[df['country'] == country]

        aad_long = (
            subset['Displaced'].sum() / n_years
        )

        print(
            f'{country}: '
            f'{subset["year"].min()}–'
            f'{subset["year"].max()}, '
            f'long-term AAD = {aad_long:,.0f}'
        )

    return df, (first, last)


# ============================================================
# LOAD IDMC DATA
# ============================================================

def read_idmc():

    df = pd.read_csv(IDMC_FILE)

    df = df[
        df['ISO3'].isin(['ETH', 'SOM', 'SDN'])
    ].copy()

    df['Year'] = pd.to_numeric(
        df['Year'],
        errors='coerce'
    )

    df['Disaster Internal Displacements'] = (
        pd.to_numeric(
            df['displacements_filtered'],
            errors='coerce'
        )
    )

    df = df.dropna(subset=['Year'])

    first = int(df['Year'].min())
    last = int(df['Year'].max())

    print('\n--- IDMC ---')
    print(f'Available years: {first}–{last}')
    print(
        f'Long-term averaging period: '
        f'{first}–{last} ({last-first+1} years)'
    )

    return df, (first, last)


# ============================================================
# MODEL AAD SUMMARY
# ============================================================

def summary(aad, country):

    data = aad[
        aad['modeler'] != 'IDMC'
    ].copy()

    if country is not None:
        data = data[
            data['country'] == country
        ]

    # Sum across countries within each model specification
    grouped = data.groupby(
        ['modeler', 'model_specification'],
        dropna=False
    )['aad'].sum()

    # Calculate variation across specifications
    return grouped.groupby(level=0).agg(
        ['mean', 'min', 'max']
    )


# ============================================================
# OBSERVATIONAL REFERENCES
# ============================================================

def observations(
    aad,
    dfo,
    idmc,
    dfo_span,
    idmc_span,
    country
):

    d = (
        dfo
        if country is None
        else dfo[dfo['country'] == country]
    )

    i = (
        idmc
        if country is None
        else idmc[idmc['ISO3'] == country]
    )

    a = aad[
        aad['modeler'] == 'IDMC'
    ]

    if country is not None:
        a = a[
            a['country'] == country
        ]

    specs = a.groupby(
        'model_specification',
        dropna=False
    )['aad'].sum()

    short_idmc = (
        specs.mean()
        if len(specs)
        else np.nan
    )

    n_study = STUDY_END - STUDY_START + 1

    short_dfo = (
        d.loc[
            d['year'].between(
                STUDY_START,
                STUDY_END
            ),
            'Displaced'
        ].sum() / n_study
    )

    long_idmc = (
        i['Disaster Internal Displacements'].sum()
        / (idmc_span[1] - idmc_span[0] + 1)
    )

    long_dfo = (
        d['Displaced'].sum()
        / (dfo_span[1] - dfo_span[0] + 1)
    )

    # Robustness checks: IDMC only. AAD denominators count calendar years,
    # including zero-displacement years, and exclude 2020 where requested.
    def idmc_aad(start, end, exclude=None):
        # Missing country-years represent zero recorded displacements.
        # Keep every calendar year in the denominator, including zero years.
        years = pd.Index(range(start, end + 1), name='Year')
        if exclude is not None:
            years = years[years != exclude]
        yearly = (
            i.groupby('Year')['Disaster Internal Displacements']
             .sum()
             .reindex(years, fill_value=0)
        )
        return yearly.sum() / len(years)

    # For the event-based panel, use the study period without 2020.
    # For probabilistic models, use the original long-term period without 2020.
    short_excl = idmc_aad(STUDY_START, STUDY_END, EXCLUDED_YEAR)
    long_excl = idmc_aad(idmc_span[0], idmc_span[1], EXCLUDED_YEAR)
    recent = idmc_aad(RECENT_START, RECENT_END)

    return {
        'IDMC robustness': {
            'Event-based': (short_excl, recent),
            'Probabilistic': (long_excl, recent)
        },
        'Event-based': (
            short_idmc,
            short_dfo
        ),
        'Probabilistic': (
            long_idmc,
            long_dfo
        )
    }


# ============================================================
# AXIS FORMATTING
# ============================================================

def format_y(value, _):

    if value >= 1e6:
        return f'{value / 1e6:g}M'

    if value >= 1e3:
        return f'{value / 1e3:g}k'

    return f'{value:g}'


# ============================================================
# DRAW MODEL PANEL
# ============================================================

def draw(
    ax,
    stats,
    refs,
    group,
    panel_big,
    periods,
    y_label=False
):

    # Model positions
    xs = {}
    x = 0.0

    for family in GROUPS[group]:

        for model in family:

            xs[model] = x
            x += FAMILY_STEP

        x += FAMILY_GAP

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    for model, x in xs.items():

        if model not in stats.index:
            print('Missing model:', model)
            continue

        row = stats.loc[model]

        mean = float(row['mean'])
        low = float(row['min'])
        high = float(row['max'])

        if not np.isfinite(mean) or mean <= 0:
            continue

        color = FAMILY_COLORS[model]
        calibrated = model in CALIBRATED

        ax.bar(
            x,
            mean,
            width=BAR_WIDTH,
            color=color,
            edgecolor='white',
            hatch='///' if calibrated else None,
            linewidth=0.75,
            alpha=1.0,
            zorder=3
        )

        # Min-max uncertainty
        if (
            np.isfinite(low)
            and np.isfinite(high)
            and low > 0
            and high >= mean >= low
        ):

            ax.errorbar(
                x,
                mean,
                yerr=[
                    [mean - low],
                    [high - mean]
                ],
                fmt='none',
                ecolor='#303030',
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
                zorder=6
            )

    # --------------------------------------------------------
    # Observational reference range
    # --------------------------------------------------------

    idmc, dfo = refs[group]
    idmc_excl, idmc_recent = refs['IDMC robustness'][group]

    # Shade the complete range of observational AAD estimates,
    # including the original IDMC and DFO references and both
    # additional IDMC robustness checks.
    observed_values = [idmc, dfo, idmc_excl, idmc_recent]
    observed_values = [
        float(v) for v in observed_values
        if np.isfinite(v) and v > 0
    ]
    if len(observed_values) >= 2:
        ax.axhspan(
            min(observed_values),
            max(observed_values),
            color='#a1a7ad',
            alpha=0.16,
            zorder=1
        )

    # --------------------------------------------------------
    # Full-width horizontal lines with left-side labels
    # --------------------------------------------------------

    for value, name, color in [
        (idmc, 'IDMC', IDMC_COLOR),
        (dfo, 'DFO', DFO_COLOR)
    ]:

        if not np.isfinite(value) or value <= 0:
            continue

        ax.axhline(
            y=value,
            xmin=0,
            xmax=1,
            color=color,
            linewidth=1.6,
            linestyle='-',
            zorder=5
        )

        # Only label lines in the large upper panel
        if panel_big:

            period = periods[name][group]

            ax.annotate(
                f'{name} ({period})',
                xy=(0.01, value),
                xycoords=(
                    'axes fraction',
                    'data'
                ),
                xytext=(0, 4),
                textcoords='offset points',
                ha='left',
                va='bottom',
                color=color,
                fontsize=11,
                weight='semibold',
                zorder=8,
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.85,
                    pad=1.5
                )
            )

    # IDMC-only robustness references (no changes to DFO or model bars).
    for value, label, color, linestyle in [
        (idmc_excl, 'IDMC (excluding 2020)', IDMC_EXCL_COLOR, '--'),
        (idmc_recent, f'IDMC ({RECENT_START}–{RECENT_END})',
         IDMC_RECENT_COLOR, ':'),
    ]:
        if not np.isfinite(value) or value <= 0:
            continue
        ax.axhline(value, color=color, linestyle=linestyle,
                   linewidth=1.8, zorder=5)
        if panel_big:
            ax.annotate(label, xy=(0.99, value),
                        xycoords=('axes fraction', 'data'),
                        xytext=(0, 4), textcoords='offset points',
                        ha='right', va='bottom', color=color,
                        fontsize=10, weight='semibold', zorder=8,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=1.5))

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.set_yscale('log')

    ax.set_xlim(
        min(xs.values()) - 0.45,
        max(xs.values()) + 0.45
    )

    ax.set_xticks(
        list(xs.values()),
        [
            NAMES.get(model, model)
            for model in xs
        ]
    )

    ax.tick_params(
        axis='x',
        rotation=48,
        length=0,
        pad=5,
        labelsize=12
    )

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment('right')
        tick.set_rotation_mode('anchor')

    ax.yaxis.set_major_locator(
        LogLocator(base=10)
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(format_y)
    )

    ax.yaxis.set_minor_formatter(
        NullFormatter()
    )

    ax.grid(
        axis='y',
        which='major',
        alpha=0.15,
        zorder=0
    )

    ax.set_axisbelow(True)

    if panel_big:
        ax.set_title(
            group,
            weight='semibold',
            pad=11
        )

    if y_label:
        ax.set_ylabel('AAD')

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


# ============================================================
# MAIN FIGURE
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    aad = pd.read_csv(AAD_FILE)

    aad['model_specification'] = (
        aad['model_specification']
        .astype(str)
        .str.replace("'", '', regex=False)
    )

    dfo, dfo_span = read_dfo()
    idmc, idmc_span = read_idmc()

    periods = {
        'IDMC': {
            'Event-based':
                f'{STUDY_START}–{STUDY_END}',
            'Probabilistic':
                f'{idmc_span[0]}–{idmc_span[1]}'
        },
        'DFO': {
            'Event-based':
                f'{STUDY_START}–{STUDY_END}',
            'Probabilistic':
                f'{dfo_span[0]}–{dfo_span[1]}'
        }
    }

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.25, 1],
        hspace=0.58,
        wspace=0.18
    )

    panels = [
        (
            gs[0, :],
            None,
            'a',
            'All countries'
        ),
        (
            gs[1, 0],
            'ETH',
            'b',
            'Ethiopia'
        ),
        (
            gs[1, 1],
            'SOM',
            'c',
            'Somalia'
        ),
        (
            gs[1, 2],
            'SDN',
            'd',
            'Sudan'
        )
    ]

    for slot, country, letter, name in panels:

        sub = slot.subgridspec(
            1,
            2,
            width_ratios=[5.2, 3.3],
            wspace=0.06
        )

        ax1 = fig.add_subplot(
            sub[0, 0]
        )

        ax2 = fig.add_subplot(
            sub[0, 1],
            sharey=ax1
        )

        stats = summary(
            aad,
            country
        )

        refs = observations(
            aad,
            dfo,
            idmc,
            dfo_span,
            idmc_span,
            country
        )

        draw(
            ax1,
            stats,
            refs,
            'Event-based',
            country is None,
            periods,
            y_label=(
                country is None
                or country == 'ETH'
            )
        )

        draw(
            ax2,
            stats,
            refs,
            'Probabilistic',
            country is None,
            periods
        )

        ax2.tick_params(
            axis='y',
            left=False,
            labelleft=False
        )

        # Panel letters and country names
        ax1.text(
            0,
            1.15,
            f'({letter})  {name}',
            transform=ax1.transAxes,
            fontsize=16,
            weight='bold',
            va='bottom'
        )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    legend = [
        Patch(
            facecolor='#7393be',
            edgecolor='white',
            label='Not calibrated'
        ),
        Patch(
            facecolor='#7393be',
            edgecolor='white',
            hatch='///',
            label='Calibrated against IDMC'
        ),
        Line2D(
            [0], [0],
            color=IDMC_COLOR,
            lw=1.8,
            label='IDMC'
        ),
        Line2D([0], [0], color=IDMC_EXCL_COLOR, lw=1.8,
               linestyle='--', label='IDMC excluding 2020'),
        Line2D([0], [0], color=IDMC_RECENT_COLOR, lw=1.8,
               linestyle=':', label='IDMC 2017–2025'),
        Line2D(
            [0], [0],
            color=DFO_COLOR,
            lw=1.8,
            label='DFO'
        ),
        Patch(
            facecolor='#a1a7ad',
            alpha=0.16,
            label='Range of observational AAD estimates'
        )
    ]

    fig.legend(
        handles=legend,
        loc='lower center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        fontsize=12,
        columnspacing=1.5,
        handlelength=1.8
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.22
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for ext in ('png', 'pdf'):

        filename = (
            OUT /
            f'aad_displacements_grouped_4panel_idmc_robustness.{ext}'
        )

        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight'
        )

        print('Saved:', filename)

    plt.close(fig)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
DISMIP annual displacement time series.

- ZIM-D: verified H08 + WaterGAP2-2e ensemble mean
- Average predictions across folds per event
- Sum events to annual country totals
- Include zero-displacement years (2008–2021)
- Spearman correlations calculated against IDMC

Figures:
1. Annual displacement (linear)
2. Annual displacement (asinh)
3. Annual displacement with ZIM-1M (linear)
4. Annual displacement with ZIM-1M (asinh)

Y-axis labels are always displayed in millions.
Scientific notation and axis offsets are disabled.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    TextArea,
    VPacker,
)
from matplotlib.ticker import (
    FuncFormatter,
    MaxNLocator,
    MultipleLocator,
    FixedLocator,
    NullLocator,
)
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

WD = Path(__file__).resolve().parents[1]

INPUT_DIR = WD / 'dismip/dismip_data'
OUTPUT_DIR = WD / 'dismip/output'

DISMIP_FILE = INPUT_DIR / 'displacements.csv'
DFO_FILE = WD / 'data/dfo_country_displacements.csv'

ZIMD_FILE = Path(__file__).resolve().parents[1] / 'data/zimd/dismip_pred_test.csv'

STANDARDIZED_DIR = WD / 'dismip/new_idmc_data'

STANDARDIZED_FILES = {
    'h08': (
        STANDARDIZED_DIR /
        'flood_event_model_inputs_standardized_h08.csv'
    ),
    'watergap2-2e': (
        STANDARDIZED_DIR /
        'flood_event_model_inputs_standardized_watergap2-2e.csv'
    ),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['ETH', 'SOM', 'SDN']
YEARS = np.arange(2008, 2022)

# Updated panel titles
PANELS = [
    ('ALL', 'a) All countries'),
    ('ETH', 'b) Ethiopia'),
    ('SOM', 'c) Somalia'),
    ('SDN', 'd) Sudan'),
]

GHMS = ['h08', 'watergap2-2e']

MATCH_PREDICTORS = [
    'ri_mean',
    'political_stability',
    'age',
    'drinking_water',
    'schooling',
    'pop_change',
]

MATCH_ATOL = 1e-6
MATCH_RTOL = 1e-6

PREDICTION_COLUMN = 'pred_median'

ZIMD_RUN = 'all'
ZIMD_RE = True

EXPECTED_FOLDS = set(range(1, 11))

SHOW_UNCERTAINTY = False

ASINH_LINEAR_WIDTH = 200_000

MARKER_SIZE = 3
LINE_WIDTH = 1.5

COLORS = {
    'RIE-0': '#6482b4',
    'RIE-FLO': '#849dc7',
    'ZIM-D': '#d69a43',
    'ZIM-1M': '#87b5a9',
    'IDMC': '#000000',
    'DFO': '#000000',
}

ORIGINAL = [
    'IDMC',
    'DFO',
    'RIE-0',
    'RIE-FLO',
    'ZIM-D',
]

WITH_ZIM1M = ORIGINAL + ['ZIM-1M']


# ============================================================
# HELPERS
# ============================================================

def require(df, cols, path):
    missing = set(cols) - set(df.columns)

    if missing:
        raise ValueError(
            f'{path}: missing columns {sorted(missing)}'
        )


def country_codes(s):
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .replace({
            'AB9': 'SDN',
            'ETHIOPIA': 'ETH',
            'SOMALIA': 'SOM',
            'SUDAN': 'SDN',
        })
    )


# ============================================================
# LOAD DISMIP
# ============================================================

def load_dismip():

    df = pd.read_csv(DISMIP_FILE)

    require(
        df,
        ['country', 'year', 'modeler', 'displacements'],
        DISMIP_FILE
    )

    if 'model_specification' not in df.columns:
        df['model_specification'] = 'default'

    df['country'] = country_codes(df['country'])
    df['year'] = pd.to_numeric(
        df['year'],
        errors='coerce'
    )

    df['displacements'] = pd.to_numeric(
        df['displacements'],
        errors='coerce'
    )

    df['modeler'] = (
        df['modeler']
        .astype(str)
        .str.strip()
    )

    df['model_specification'] = (
        df['model_specification']
        .astype(str)
        .str.replace("'", '', regex=False)
    )

    df = df[
        df.country.isin(COUNTRIES)
        & df.year.isin(YEARS)
    ].dropna(
        subset=['year', 'displacements']
    ).copy()

    df['year'] = df['year'].astype(int)

    # Model predictions from existing DISMIP file
    models = df[
        df.modeler.isin([
            'RIE-0',
            'RIE-FLO',
            'ZIM-1M'
        ])
    ]

    per_spec = (
        models.groupby(
            [
                'modeler',
                'model_specification',
                'country',
                'year',
            ],
            as_index=False
        )['displacements']
        .sum()
    )

    summary = (
        per_spec.groupby(
            ['modeler', 'country', 'year'],
            as_index=False
        )
        .agg(
            value=('displacements', 'mean'),
            lower=('displacements', 'min'),
            upper=('displacements', 'max'),
        )
    )

    # IDMC observations
    idmc = (
        df[df.modeler.eq('IDMC')]
        .groupby(
            ['country', 'year'],
            as_index=False
        )['displacements']
        .sum()
        .rename(
            columns={'displacements': 'value'}
        )
    )

    idmc['modeler'] = 'IDMC'
    idmc['lower'] = idmc['value']
    idmc['upper'] = idmc['value']

    return pd.concat(
        [summary, idmc],
        ignore_index=True
    )


# ============================================================
# LOAD DFO
# ============================================================

def load_dfo():

    df = pd.read_csv(DFO_FILE)

    require(
        df,
        [
            'Country',
            'Began',
            'Displaced_original'
        ],
        DFO_FILE
    )

    df['country'] = country_codes(df['Country'])

    df['year'] = pd.to_numeric(
        df['Began'],
        errors='coerce'
    )

    df['value'] = pd.to_numeric(
        df['Displaced_original'],
        errors='coerce'
    )

    df = df[
        df.country.isin(COUNTRIES)
        & df.year.isin(YEARS)
    ].copy()

    df = (
        df.groupby(
            ['country', 'year'],
            as_index=False
        )['value']
        .sum()
    )

    df['modeler'] = 'DFO'
    df['lower'] = df['value']
    df['upper'] = df['value']

    return df


# ============================================================
# VERIFY STANDARDIZED ZIM-D PREDICTIONS
# ============================================================

def verify_standardized_predictions(df):

    keys = [
        'event_id',
        'country',
        'year',
        'ghm'
    ]

    prediction_keys = keys + [
        'fold',
        're'
    ]

    require(
        df,
        keys + MATCH_PREDICTORS,
        ZIMD_FILE
    )

    reference_frames = []

    for ghm in GHMS:

        path = STANDARDIZED_FILES[ghm]
        source = pd.read_csv(path)

        if (
            'ISO3' not in source.columns
            and 'iso3' in source.columns
        ):
            source = source.rename(
                columns={'iso3': 'ISO3'}
            )

        require(
            source,
            [
                'event_id',
                'ISO3',
                'year'
            ] + MATCH_PREDICTORS,
            path
        )

        source['country'] = country_codes(
            source['ISO3']
        )

        source['ghm'] = ghm

        source['year'] = pd.to_numeric(
            source['year'],
            errors='coerce'
        )

        source['event_id'] = (
            source['event_id']
            .astype('string')
            .str.strip()
        )

        source = source[
            source.country.isin(COUNTRIES)
            & source.year.isin(YEARS)
        ].copy()

        # Mirror prediction-time substitutions
        source['ri_mean'] = pd.to_numeric(
            source['ri_mean'],
            errors='coerce'
        )

        source.loc[
            ~np.isfinite(source['ri_mean']),
            'ri_mean'
        ] = 1.0

        if 'pop_affected' in source.columns:

            source['pop_affected'] = pd.to_numeric(
                source['pop_affected'],
                errors='coerce'
            )

            source.loc[
                (
                    ~np.isfinite(
                        source['pop_affected']
                    )
                )
                | (
                    source['pop_affected'] < 1
                ),
                'pop_affected'
            ] = 1.0

        reference_frames.append(source)

    reference = pd.concat(
        reference_frames,
        ignore_index=True
    )

    compare_cols = list(MATCH_PREDICTORS)

    if (
        'pop_affected' in df.columns
        and 'pop_affected' in reference.columns
    ):
        compare_cols.append('pop_affected')

    reference = (
        reference[keys + compare_cols]
        .drop_duplicates()
        .copy()
    )

    predictions = (
        df.reset_index(drop=True).copy()
    )

    predictions['_prediction_row_id'] = (
        np.arange(len(predictions))
    )

    merged = predictions.merge(
        reference,
        on=keys,
        how='left',
        suffixes=('', '_reference'),
        indicator=True
    )

    matched = (
        merged['_merge']
        .eq('both')
        .to_numpy()
    )

    for col in compare_cols:

        actual = pd.to_numeric(
            merged[col],
            errors='coerce'
        ).to_numpy(dtype=float)

        expected = pd.to_numeric(
            merged[f'{col}_reference'],
            errors='coerce'
        ).to_numpy(dtype=float)

        matched &= np.isclose(
            actual,
            expected,
            atol=MATCH_ATOL,
            rtol=MATCH_RTOL,
            equal_nan=False
        )

    valid_ids = pd.Index(
        merged.loc[
            matched,
            '_prediction_row_id'
        ].unique()
    )

    is_verified = (
        predictions['_prediction_row_id']
        .isin(valid_ids)
    )

    verified = (
        predictions.loc[is_verified]
        .drop(
            columns='_prediction_row_id'
        )
        .copy()
    )

    rejected = (
        predictions.loc[~is_verified]
        .drop(
            columns='_prediction_row_id'
        )
        .copy()
    )

    rejected.to_csv(
        OUTPUT_DIR /
        'zimd_rejected_unverified_predictions.csv',
        index=False
    )

    print(
        f'Prediction verification: '
        f'{len(verified)} verified, '
        f'{len(rejected)} rejected.'
    )

    if verified.empty:
        raise ValueError(
            'No predictions match standardized inputs.'
        )

    n_before = len(verified)

    verified = verified.drop_duplicates(
        subset=prediction_keys,
        keep='first'
    ).copy()

    print(
        f'Removed {n_before - len(verified)} '
        f'duplicate verified predictions.'
    )

    verified.to_csv(
        OUTPUT_DIR /
        'zimd_verified_standardized_predictions.csv',
        index=False
    )

    return verified


# ============================================================
# LOAD ZIM-D ENSEMBLE
# ============================================================

def load_zimd():

    df = pd.read_csv(ZIMD_FILE)

    if (
        'ghm' not in df.columns
        and 'GHM' in df.columns
    ):
        df = df.rename(
            columns={'GHM': 'ghm'}
        )

    if (
        'ISO3' not in df.columns
        and 'iso3' in df.columns
    ):
        df = df.rename(
            columns={'iso3': 'ISO3'}
        )

    require(
        df,
        [
            'event_id',
            'ISO3',
            'year',
            'ghm',
            'fold',
            're',
            PREDICTION_COLUMN,
        ] + MATCH_PREDICTORS,
        ZIMD_FILE
    )

    df['country'] = country_codes(
        df['ISO3']
    )

    df['year'] = pd.to_numeric(
        df['year'],
        errors='coerce'
    )

    df['fold'] = pd.to_numeric(
        df['fold'],
        errors='coerce'
    )

    df[PREDICTION_COLUMN] = pd.to_numeric(
        df[PREDICTION_COLUMN],
        errors='coerce'
    )

    df['ghm'] = (
        df['ghm']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df['event_id'] = (
        df['event_id']
        .astype('string')
        .str.strip()
    )

    mask = (
        df.country.isin(COUNTRIES)
        & df.year.isin(YEARS)
        & df['ghm'].isin(GHMS)
        & df['re']
        .astype(str)
        .str.lower()
        .eq(str(ZIMD_RE).lower())
    )

    if 'run' in df.columns:

        mask &= (
            df['run']
            .astype(str)
            .str.strip()
            .eq(ZIMD_RUN)
        )

    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError(
            'No ZIM-D predictions match the filters.'
        )

    if df[
        ['event_id', 'year', 'fold']
    ].isna().any().any():

        raise ValueError(
            'Missing event ID, year or fold.'
        )

    # Verify against standardized inputs
    df = verify_standardized_predictions(df)

    if not np.isfinite(
        df[PREDICTION_COLUMN]
    ).all():

        raise ValueError(
            'Invalid verified ZIM-D predictions.'
        )

    event_keys = [
        'ghm',
        'country',
        'year',
        'event_id',
    ]

    # Check 10-fold coverage
    fold_counts = (
        df.groupby(event_keys)['fold']
        .agg(
            lambda x: set(
                x.astype(int)
            )
        )
    )

    incomplete = fold_counts[
        fold_counts.map(
            lambda x: x != EXPECTED_FOLDS
        )
    ]

    if len(incomplete):

        print(
            f'WARNING: {len(incomplete)} '
            f'event/GHM combinations '
            f'lack all 10 folds.'
        )

        print(
            incomplete.head(10).to_string()
        )

        raise ValueError(
            'Incomplete fold coverage.'
        )

    # Average across folds for each event
    events = (
        df.groupby(
            event_keys,
            as_index=False
        )[PREDICTION_COLUMN]
        .mean()
        .rename(
            columns={
                PREDICTION_COLUMN:
                'event_prediction'
            }
        )
    )

    events.to_csv(
        OUTPUT_DIR /
        'zimd_event_predictions_fold_mean.csv',
        index=False
    )

    # Sum events to annual totals per GHM
    annual = (
        events.groupby(
            [
                'ghm',
                'country',
                'year'
            ],
            as_index=False
        )['event_prediction']
        .sum(min_count=1)
        .rename(
            columns={
                'event_prediction': 'value'
            }
        )
    )

    # Add zero-displacement years
    full_index = (
        pd.MultiIndex.from_product(
            [
                GHMS,
                COUNTRIES,
                YEARS
            ],
            names=[
                'ghm',
                'country',
                'year'
            ]
        )
    )

    annual = (
        annual
        .set_index([
            'ghm',
            'country',
            'year'
        ])
        .reindex(full_index)
        .reset_index()
    )

    zero_years = (
        annual['value'].isna().sum()
    )

    print(
        f'Added {zero_years} '
        f'GHM/country-years '
        f'with zero displacement.'
    )

    annual['value'] = (
        annual['value'].fillna(0.0)
    )

    annual.to_csv(
        OUTPUT_DIR /
        'zimd_annual_by_ghm.csv',
        index=False
    )

    # H08 + WaterGAP2-2e ensemble mean
    ensemble = (
        annual.groupby(
            ['country', 'year'],
            as_index=False
        )['value']
        .mean()
    )

    ensemble['modeler'] = 'ZIM-D'
    ensemble['lower'] = ensemble['value']
    ensemble['upper'] = ensemble['value']

    ensemble.to_csv(
        OUTPUT_DIR /
        'zimd_annual_ensemble_mean.csv',
        index=False
    )

    return ensemble[
        [
            'country',
            'year',
            'modeler',
            'value',
            'lower',
            'upper',
        ]
    ]


# ============================================================
# COMBINE DATA
# ============================================================

def load_all():

    data = pd.concat(
        [
            load_dismip(),
            load_dfo(),
            load_zimd(),
        ],
        ignore_index=True
    )

    # Add missing observation years as zero
    obs_idx = (
        pd.MultiIndex.from_product(
            [
                ['IDMC', 'DFO'],
                COUNTRIES,
                YEARS,
            ],
            names=[
                'modeler',
                'country',
                'year'
            ]
        )
    )

    obs = (
        data[
            data.modeler.isin([
                'IDMC',
                'DFO'
            ])
        ]
        .set_index([
            'modeler',
            'country',
            'year'
        ])
        .reindex(obs_idx)
        .reset_index()
    )

    for col in [
        'value',
        'lower',
        'upper'
    ]:
        obs[col] = obs[col].fillna(0)

    pred = data[
        ~data.modeler.isin([
            'IDMC',
            'DFO'
        ])
    ]

    data = pd.concat(
        [obs, pred],
        ignore_index=True
    )

    # Pool ETH, SOM and SDN for ALL panel
    pooled = (
        data.groupby(
            ['modeler', 'year'],
            as_index=False
        )[
            ['value', 'lower', 'upper']
        ]
        .sum(min_count=1)
    )

    pooled['country'] = 'ALL'

    return pd.concat(
        [data, pooled],
        ignore_index=True
    )


# ============================================================
# SPEARMAN CORRELATIONS AGAINST IDMC
# ============================================================

def stars(p):

    if p < 0.001:
        return '***'

    if p < 0.01:
        return '**'

    if p < 0.05:
        return '*'

    return ''


def correlations(data, country, models):

    obs = (
        data[
            (data.country == country)
            & data.modeler.eq('IDMC')
        ][['year', 'value']]
        .rename(
            columns={'value': 'obs'}
        )
    )

    out = []

    for model in models:

        pred = (
            data[
                (data.country == country)
                & data.modeler.eq(model)
            ][['year', 'value']]
            .rename(
                columns={'value': 'pred'}
            )
        )

        paired = (
            pred.merge(
                obs,
                on='year'
            )
            .dropna()
        )

        if (
            len(paired) >= 3
            and paired.pred.nunique() > 1
            and paired.obs.nunique() > 1
        ):

            rho, p = spearmanr(
                paired.pred,
                paired.obs
            )

            label = (
                f'{model}: '
                f'ρ = {rho:.2f}{stars(p)}'
            )

        else:

            label = (
                f'{model}: ρ = N/A'
            )

        out.append(
            (model, label)
        )

    return out


# ============================================================
# AXIS HELPERS
# ============================================================

def values_for_panel(
    data,
    country,
    models
):

    d = data[
        (data.country == country)
        & data.modeler.isin(models)
    ]

    v = (
        d[
            ['value', 'lower', 'upper']
        ]
        .to_numpy(dtype=float)
        .ravel()
    )

    return np.sort(
        np.unique(
            v[
                np.isfinite(v)
                & (v >= 0)
            ]
        )
    )


def asinh_forward(x):

    return np.arcsinh(
        np.asarray(
            x,
            dtype=float
        ) / ASINH_LINEAR_WIDTH
    )


def asinh_inverse(x):

    return (
        ASINH_LINEAR_WIDTH
        * np.sinh(
            np.asarray(
                x,
                dtype=float
            )
        )
    )


def axis_ticks(maximum):

    candidates = np.array([
        0,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
        200_000_000,
        500_000_000,
    ], dtype=float)

    return candidates[
        candidates <= maximum * 1.001
    ]


# ============================================================
# PLOT SERIES
# ============================================================

def plot_series(
    ax,
    data,
    country,
    models
):

    subset = data[
        data.country.eq(country)
    ]

    for model in models:

        s = (
            subset[
                subset.modeler.eq(model)
            ]
            .set_index('year')
            .reindex(YEARS)
        )

        y = (
            s.value
            .to_numpy(dtype=float)
        )

        linestyle = (
            '--'
            if model == 'DFO'
            else '-'
        )

        ax.plot(
            YEARS,
            y,
            color=COLORS[model],
            ls=linestyle,
            lw=LINE_WIDTH,
            marker='o',
            markersize=MARKER_SIZE,
            zorder=3
        )

        # No shading for ZIM-1M
        if (
            SHOW_UNCERTAINTY
            and model in [
                'RIE-0',
                'RIE-FLO'
            ]
        ):

            ax.fill_between(
                YEARS,
                s.lower.to_numpy(
                    dtype=float
                ),
                s.upper.to_numpy(
                    dtype=float
                ),
                color=COLORS[model],
                alpha=0.18,
                linewidth=0,
                zorder=2
            )


# ============================================================
# FORMAT AXIS
# ============================================================

def format_axis(
    ax,
    bottom=False
):

    ax.set_xlim(
        YEARS[0] - 0.3,
        YEARS[-1] + 0.3
    )

    ax.set_xticks(
        np.arange(
            2008,
            2022,
            2
        )
    )

    # Force y-axis labels into millions
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position:
            f'{value / 1e6:g}'
        )
    )

    # Disable scientific notation / offsets
    ax.yaxis.get_offset_text().set_visible(
        False
    )

    ax.tick_params(
        labelsize=9
    )

    ax.grid(
        axis='y',
        alpha=0.18
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if bottom:

        ax.tick_params(
            axis='x',
            labelrotation=45
        )

    else:

        ax.tick_params(
            axis='x',
            bottom=False,
            labelbottom=False
        )


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel(
    fig,
    slot,
    data,
    country,
    title,
    models,
    transformed
):

    ax = fig.add_subplot(slot)

    vals = values_for_panel(
        data,
        country,
        models
    )

    plot_series(
        ax,
        data,
        country,
        models
    )

    maximum = (
        max(
            vals[-1] * 1.08,
            1
        )
        if len(vals)
        else 1
    )

    # --------------------------------------------------------
    # Y-AXIS SCALING
    # --------------------------------------------------------

    if transformed:

        ax.set_yscale(
            'function',
            functions=(
                asinh_forward,
                asinh_inverse
            )
        )

        ticks = axis_ticks(
            maximum
        )

        ax.yaxis.set_major_locator(
            FixedLocator(ticks)
        )

        ax.yaxis.set_minor_locator(
            NullLocator()
        )

    else:

        if country == 'ALL':

            # All countries: 5-million intervals
            ax.yaxis.set_major_locator(
                MultipleLocator(
                    5_000_000
                )
            )

        else:

            # Adaptive limits for each country
            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=5,
                    min_n_ticks=3
                )
            )

    ax.set_ylim(
        0,
        maximum
    )

    format_axis(
        ax,
        bottom=True
    )

    # Explicitly enforce millions
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position:
            f'{value / 1_000_000:g}'
        )
    )

    ax.yaxis.get_offset_text().set_visible(
        False
    )

    # --------------------------------------------------------
    # PANEL TITLE: UPPER LEFT INSIDE AXES
    # --------------------------------------------------------

    ax.set_title('')

    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        fontweight='bold',
        zorder=12,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.75,
            pad=1.5
        )
    )

    # --------------------------------------------------------
    # SPEARMAN CORRELATIONS AGAINST IDMC
    # --------------------------------------------------------

    annotations = [
        (m, t)
        for m, t in correlations(
            data,
            country,
            models
        )
        if m not in (
            'IDMC',
            'DFO'
        )
    ]

    if annotations:

        lines = [
            TextArea(
                label,
                textprops={
                    'color': COLORS[model],
                    'fontsize': (
                        8
                        if country != 'ALL'
                        else 9
                    ),
                }
            )
            for model, label in annotations
        ]

        content = VPacker(
            children=lines,
            align='left',
            pad=0,
            sep=3
        )

        box = AnchoredOffsetbox(
            loc='upper left',
            child=content,
            frameon=True,

            # Moved downward to avoid title
            bbox_to_anchor=(
                0.02,
                0.86
            ),

            bbox_transform=ax.transAxes,
            borderpad=0,
            pad=0.4
        )

        box.patch.set(
            facecolor='white',
            edgecolor='0.75',
            linewidth=0.6,
            alpha=0.88
        )

        box.set_zorder(10)
        ax.add_artist(box)

    if country in (
        'ALL',
        'ETH'
    ):

        ax.set_ylabel(
            'Displacements (millions)',
            fontsize=10
        )


# ============================================================
# LEGEND
# ============================================================

def legend(
    fig,
    models
):

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            lw=LINE_WIDTH,
            ls=(
                '--'
                if m == 'DFO'
                else '-'
            ),
            marker='o',
            markersize=5,
            label=m
        )
        for m in models
    ]

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(models),
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.015
        ),
        fontsize=10
    )


# ============================================================
# MAKE ANNUAL DISPLACEMENT FIGURES
# ============================================================

def make_figure(
    data,
    models,
    filename,
    transformed=False
):

    fig = plt.figure(
        figsize=(10, 6)
    )

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[
            1.2,
            1
        ],
        left=0.09,
        right=0.98,
        top=0.94,
        bottom=0.14,
        hspace=0.43,
        wspace=0.34
    )

    slots = [
        gs[0, :],
        gs[1, 0],
        gs[1, 1],
        gs[1, 2],
    ]

    for (
        country,
        title
    ), slot in zip(
        PANELS,
        slots
    ):

        draw_panel(
            fig,
            slot,
            data,
            country,
            title,
            models,
            transformed
        )

    legend(
        fig,
        models
    )

    suffix = (
        '_asinh'
        if transformed
        else ''
    )

    for ext in [
        'png',
        'pdf'
    ]:

        fig.savefig(
            OUTPUT_DIR /
            f'{filename}{suffix}.{ext}',
            dpi=300,
            bbox_inches='tight'
        )

    plt.close(fig)

    print(
        'Saved:',
        filename + suffix
    )


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_all()

    for transformed in (
        False,
        True
    ):

        # Main DISMIP comparison
        make_figure(
            data,
            ORIGINAL,
            'annual_change_four_panel_updated_ZIMD',
            transformed
        )

        # Additional comparison including ZIM-1M
        make_figure(
            data,
            WITH_ZIM1M,
            'annual_change_four_panel_updated_ZIMD_with_ZIM1M',
            transformed
        )

    print('Complete.')


if __name__ == '__main__':
    main()

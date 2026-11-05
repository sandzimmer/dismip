#!/usr/bin/env python3
from pathlib import Path
# !/usr/bin/env python3

### Import Modules ###

# Make paths
wd = str(Path(__file__).resolve().parents[1]) + '/'
output_path = wd + 'output/'
dismip_path = wd + 'dismip/'
input_path = dismip_path + 'dismip_data/'

import sys
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import fiona
import os
import glob
from statistics import mean, median
from datetime import datetime
import cf_xarray
import rasterio
import regionmask
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib.colors import LogNorm
import itertools


# import gdal

########################################################################################################################

## initialize final dataframes

# aad
aad_df = pd.DataFrame()

# yearly values
displacements = pd.DataFrame()

# sums (2008-2021)
displacement_sums = pd.DataFrame()

exposure = pd.DataFrame()

aae_df = pd.DataFrame()

exposure_sums = pd.DataFrame()


########## read in Zimmermann et al.

zimmermann = pd.read_csv(input_path + 'dismip_zimmermann_final2.csv')
zimmermann = zimmermann[zimmermann['level'] == 0]
print(zimmermann)
zimmermann = zimmermann[['country', 'year', 'displacements', 'idmc', 'GHM', 'protection']]

zimmermann['modeler'] = 'ZIM'
zimmermann['model_specification'] = zimmermann['GHM'] + '+' + zimmermann['protection']

years = range(2008,2022)
countries = ['ETH', 'SOM', 'SDN']



'''for country in countries:
        zimmermann_sub = zimmermann[zimmermann['country'] == country]

        country_years = np.unique(zimmermann_sub['year'])

        missing_years = [year for year in years if year not in country_years]

        for year in missing_years:
                zimmermann_sub_year = zimmermann_sub[zimmermann_sub['year'] == 2020]
                zimmermann_sub_year['year'] = year
                zimmermann_sub_year['displacements'] = 0
                zimmermann_sub_year['idmc'] = 0

                zimmermann = pd.concat([zimmermann, zimmermann_sub_year])'''


idmc = zimmermann.copy()
idmc = idmc[['country', 'year', 'modeler', 'model_specification', 'idmc']]
idmc = idmc.rename(columns = {'idmc': 'displacements'})
idmc['modeler'] = 'IDMC'
idmc['model_specification'] = '-'
idmc = idmc.drop_duplicates()

print('this is the sum of idmc')
print(idmc['displacements'].sum())


# Use the full 2008–2021 window, including zero-event years.
years = range(2008, 2022)
countries = ['ETH', 'SOM', 'SDN']

# ------------------------------------------------------------
# NEW ZIM-D: out-of-sample event predictions and uncertainty
# ------------------------------------------------------------
ZIMD_PRED_FILE = str(Path(__file__).resolve().parents[1] / 'data/zimd/dismip_pred_test.csv')
GHMS = ['h08', 'watergap2-2e']

pred = pd.read_csv(ZIMD_PRED_FILE)
pred.columns = pred.columns.str.strip()
if 'ghm' not in pred.columns and 'GHM' in pred.columns:
    pred = pred.rename(columns={'GHM': 'ghm'})
if 'ISO3' not in pred.columns and 'iso3' in pred.columns:
    pred = pred.rename(columns={'iso3': 'ISO3'})

# The R export may use slightly different names for the lower CI.
lower_candidates = ['lower_CI', 'lower_ci', 'ci_lower', 'lower', 'pred_lower']
upper_candidates = ['upper_CI', 'upper_ci', 'ci_upper', 'upper', 'pred_upper']
lower_col = next((c for c in lower_candidates if c in pred.columns), None)
upper_col = next((c for c in upper_candidates if c in pred.columns), None)
if lower_col is None or upper_col is None:
    raise ValueError('Both lower and upper prediction CI columns are required. '
                     f'Available columns: {pred.columns.tolist()}')

required = ['event_id', 'ISO3', 'year', 'ghm', 'fold', 're',
            'pred_median', lower_col, upper_col]
missing = [c for c in required if c not in pred.columns]
if missing:
    raise ValueError(f'Missing ZIM-D columns: {missing}')

pred['country'] = pred['ISO3'].astype(str).str.strip().str.upper()
pred['ghm'] = pred['ghm'].astype(str).str.strip().str.lower()
pred['event_id'] = pred['event_id'].astype('string').str.strip()
pred['year'] = pd.to_numeric(pred['year'], errors='coerce')
pred['fold'] = pd.to_numeric(pred['fold'], errors='coerce')
for col in ['pred_median', lower_col, upper_col]:
    pred[col] = pd.to_numeric(pred[col], errors='coerce')

mask = (pred['country'].isin(countries) & pred['year'].isin(years)
        & pred['ghm'].isin(GHMS)
        & pred['re'].astype(str).str.lower().eq('true'))
if 'run' in pred.columns:
    mask &= pred['run'].astype(str).str.strip().eq('all')
pred = pred.loc[mask].copy()
if pred.empty:
    raise ValueError('No ZIM-D predictions remain after filtering.')
if pred[required].isna().any().any():
    raise ValueError('Missing required values in filtered ZIM-D predictions.')

# Remove repeated exports before fold averaging.
fold_keys = ['event_id', 'country', 'year', 'ghm', 'fold', 're']
old_n = len(pred)
pred = pred.drop_duplicates(subset=fold_keys, keep='first').copy()
print(f'ZIM-D: dropped {old_n - len(pred)} duplicate prediction rows')

# Mean of fold-level estimates for each event, separately for
# central prediction, lower CI and upper CI.
event_keys = ['event_id', 'country', 'year', 'ghm']
fold_sets = pred.groupby(event_keys)['fold'].agg(lambda x: set(x.astype(int)))
incomplete = fold_sets[fold_sets.map(lambda x: x != set(range(1, 11)))]
if len(incomplete):
    print('WARNING: ZIM-D event/GHM groups with incomplete folds:', len(incomplete))
    print(incomplete.head(10).to_string())
    raise ValueError('Cannot calculate full 10-fold event means.')

events = pred.groupby(event_keys, as_index=False).agg(
    central=('pred_median', 'mean'),
    lower=(lower_col, 'mean'),
    upper=(upper_col, 'mean')
)
if ((events['lower'] > events['central']) |
    (events['central'] > events['upper'])).any():
    raise ValueError('Some averaged prediction CIs do not contain their central estimate.')

# Annual totals: sum event estimates and event CI endpoints.
annual = events.groupby(['ghm', 'country', 'year'], as_index=False).agg(
    central=('central', 'sum'), lower=('lower', 'sum'), upper=('upper', 'sum')
)

# Reindex ALL GHM/country/year combinations, so missing event years = 0.
full = pd.MultiIndex.from_product(
    [GHMS, countries, years], names=['ghm', 'country', 'year']
)
annual = annual.set_index(['ghm', 'country', 'year']).reindex(full)
zero_count = annual['central'].isna().sum()
annual[['central', 'lower', 'upper']] = annual[
    ['central', 'lower', 'upper']].fillna(0.0)
annual = annual.reset_index()
print(f'ZIM-D: added {zero_count} zero-event GHM/country-years')

# Central estimate = mean of both GHM annual totals.
# Uncertainty envelope = smallest lower CI and largest upper CI
# among the GHMs, capturing both model and prediction uncertainty.
ensemble = annual.groupby(['country', 'year'], as_index=False).agg(
    central=('central', 'mean'),
    lower=('lower', 'min'),
    upper=('upper', 'max')
)

# Save audit tables.
events.to_csv(input_path + 'zimd_new_event_predictions.csv', index=False)
annual.to_csv(input_path + 'zimd_new_annual_by_ghm.csv', index=False)
ensemble.to_csv(input_path + 'zimd_new_annual_ensemble_bounds.csv', index=False)

# Export three model specifications so the existing DISMIP
# pipeline can retain the mean and both uncertainty endpoints.
zimmermann = ensemble.melt(
    id_vars=['country', 'year'],
    value_vars=['central', 'lower', 'upper'],
    var_name='model_specification', value_name='displacements'
)
zimmermann['model_specification'] = zimmermann[
    'model_specification'].replace({'central': 'mean'})
zimmermann['modeler'] = 'ZIM-D'

# Each specification has exactly 14 country-year values.
assert (zimmermann.groupby(['country', 'model_specification'])['year']
        .nunique() == len(years)).all()

sums = zimmermann.groupby(
    ['country', 'model_specification', 'modeler'], as_index=False
)['displacements'].sum().rename(columns={'displacements': 'sum_displacements'})

aad = zimmermann.groupby(
    ['country', 'model_specification', 'modeler'], as_index=False
)['displacements'].mean().rename(columns={'displacements': 'aad'})

# IDMC AAD must also divide by all 14 years.
idmc = idmc[idmc['country'].isin(countries) & idmc['year'].isin(years)].copy()
idmc = idmc.groupby(['country', 'year'], as_index=False)['displacements'].sum()
idmc = (idmc.set_index(['country', 'year']).reindex(
    pd.MultiIndex.from_product([countries, years], names=['country', 'year']),
    fill_value=0).reset_index())
idmc['modeler'] = 'IDMC'
idmc['model_specification'] = '-'

aad_idmc = idmc.groupby(
    ['country', 'model_specification', 'modeler'], as_index=False
)['displacements'].mean().rename(columns={'displacements': 'aad'})
sums_idmc = idmc.groupby(
    ['country', 'model_specification', 'modeler'], as_index=False
)['displacements'].sum().rename(columns={'displacements': 'sum_displacements'})

aad = pd.concat([aad, aad_idmc], ignore_index=True)
sums = pd.concat([sums, sums_idmc], ignore_index=True)
displacements = pd.concat([zimmermann, idmc], ignore_index=True)
print('ZIM-D ensemble AAD (mean, lower, upper):')
print(aad[aad['modeler'].eq('ZIM-D')].to_string(index=False))

#######################################################################################################################
### add idmc data


######################################################################################################################

zimmermann = pd.read_csv(input_path + 'dismip_data_1m_zimmermann.csv')
zimmermann['modeler'] = 'ZIM-1M'
zimmermann['model_specification'] = zimmermann['GHM'] + '+' + zimmermann['protection']
zimmermann = zimmermann.rename(columns = {'pop_exposed': 'displacements', 'ISO3':'country'})
zimmermann = zimmermann[['modeler', 'country', 'model_specification', 'displacements', 'year']]

zim_sums = zimmermann.groupby(['country','model_specification','modeler']).sum()


zim_sums = zim_sums.reset_index(drop = False)
zim_sums = zim_sums.rename(columns = {'displacements':'sum_displacements'})
print(zim_sums)
#zim_sums = zim_sums.drop(columns = ['year'])
zim_sums = zim_sums.drop_duplicates()



zim_aad = zimmermann.groupby(['country','model_specification','modeler']).mean()
zim_aad = zim_aad.reset_index(drop = False)
zim_aad = zim_aad.rename(columns = {'displacements':'aad'})
#zim_aad = zim_aad.drop(columns = 'year')

displacements = pd.concat([displacements, zimmermann])

aad = pd.concat([aad, zim_aad])

print(aad['modeler'])

sums = pd.concat([sums, zim_sums])


######## read in Meiler et al ##########################################################################################
meiler = pd.read_excel(input_path + 'dismip_meiler_final.xlsx')
meiler['model_specification'] = meiler['model'] + '+' + meiler['threshold'].astype(str)
meiler['modeler'] = np.where(meiler['model'] == 'IVM', 'MEI-IVM', 'MEI-CAPRA')
meiler = meiler.drop(columns = ['model'])


sums_meiler = meiler
sums_meiler['sum_displacements'] = sums_meiler['aad'] * len(years)
sums_meiler = sums_meiler.drop(columns = ['aad'])

## add meiler to aad

print('this is sums before meiler is added')
print(sums)

aad = pd.concat([aad, meiler], axis = 0)
sums = pd.concat([sums, sums_meiler], axis = 0)



########### read in riedels data
riedel_aad = pd.read_csv(input_path + 'dismip_riedel_aad.csv')
riedel_aad = riedel_aad.rename(columns={'Year': 'year', 'Country': 'country', 'AAD':'aad'})

riedel_aad['modeler'] = np.where(riedel_aad['Protection'] == 'No Protection', 'RIE-0', 'RIE-FLO')
riedel_aad['Impact Func ID'] = riedel_aad['Impact Func ID'].astype(str)
riedel_aad['model_specification'] = riedel_aad['Protection'] + '+' + riedel_aad['Impact Func ID']

riedel_aad = riedel_aad.drop(columns = ['Protection', 'Impact Func ID'])


aad = pd.concat([aad, riedel_aad], axis = 0)

riedel_displacements = pd.read_csv(input_path + 'dismip_riedel.csv')
riedel_displacements = riedel_displacements.rename(columns={'Year': 'year', 'Country': 'country', 'Impact':'displacements'})

riedel_displacements['modeler'] = np.where(riedel_displacements['Protection'] == 'No Protection', 'RIE-0', 'RIE-FLO')
riedel_displacements['Impact Func ID'] = riedel_displacements['Impact Func ID'].astype(str)

riedel_displacements['model_specification'] = riedel_displacements['Protection'] + '+' + riedel_displacements['Impact Func ID']
riedel_displacements = riedel_displacements[riedel_displacements['year'].isin(years)]
riedel_displacements = riedel_displacements.drop(columns = ['Protection', 'Impact Func ID'])

displacements = pd.concat([displacements, riedel_displacements], axis = 0)

sums_riedel = riedel_displacements.groupby(['country','model_specification','modeler']).sum()
sums_riedel = sums_riedel.reset_index(drop = False)
sums_riedel = sums_riedel.rename(columns = {'displacements':'sum_displacements'})


sums = pd.concat([sums, sums_riedel], axis = 0)


######### read in Kams data
kam = pd.read_excel(input_path + 'dismip_kam_final.xlsx')
kam['modeler'] = 'KAM'
kam['model_specification'] = kam['rcp'] + '+' + kam['model']
kam = kam.drop(columns = ['rcp', 'model', 'pop_exposed'])

kam = kam.rename(columns = {'pop_displaced':'displacements'})

print(kam)
displacements = pd.concat([displacements, kam], axis = 0)
print(displacements)

sums_kam = kam.groupby(['country','model_specification','modeler']).sum()


sums_kam = sums_kam.reset_index(drop = False)
sums_kam = sums_kam.rename(columns = {'displacements':'sum_displacements'})
#sums = sums.drop(columns = ['year'])
sums_kam = sums_kam.drop_duplicates()


aad_kam = kam.groupby(['country','model_specification','modeler']).mean()
aad_kam = aad_kam.reset_index(drop = False)
aad_kam = aad_kam.rename(columns = {'displacements':'aad'})
#aad = aad.drop(columns = 'year')

aad = pd.concat([aad, aad_kam], axis = 0)
sums = pd.concat([sums, sums_kam], axis = 0)



####### read in CIMAS data
cima = pd.read_excel(input_path + 'dismip_cima.xlsx')
cima['modeler'] = 'CIMA'
cima['model_specification'] = '-'


sums_cima = cima
sums_cima['sum_displacements'] = sums_cima['aad'] * len(years)
sums_cima = sums_cima.drop(columns = ['aad'])


aad = pd.concat([aad, cima], axis = 0)
print(aad['modeler'].unique())
sums = pd.concat([sums, sums_cima], axis = 0)
print(sums['modeler'].unique())
print(displacements['modeler'].unique())


displacements.to_csv(input_path + 'displacements.csv')
sums.to_csv(input_path + 'sum_displacements.csv')
aad.to_csv(input_path + 'aad.csv')

########################################################################################################################

                            ### exposure ###

########################################################################################################################



zimmermann = pd.read_csv(input_path + 'dismip_zimmermann_exposure.csv')
zimmermann.rename(columns = {'ISO3':'country'}, inplace = True)
zimmermann['modeler'] = 'ZIM'
zimmermann['model_specification'] = zimmermann['GHM'] + '+' + zimmermann['protection']

print(zimmermann)

zimmermann = zimmermann[['country','model_specification','modeler', 'pop_exposed', 'year']]

sums = zimmermann.groupby(['country','model_specification','modeler']).sum()


sums = sums.reset_index(drop = False)
sums = sums.rename(columns = {'pop_exposed':'sum_exposed'})
sums = sums.drop(columns = ['year'])
sums = sums.drop_duplicates()


print(zimmermann.groupby(['country','model_specification','modeler']))
aae = zimmermann.groupby(['country','model_specification','modeler']).mean()
aae = aae.reset_index(drop = False)
aae = aae.rename(columns = {'pop_exposed':'aae'})
aae = aae.drop(columns = 'year')


aae_df = pd.concat([aae_df, aae])
exposure_sums = pd.concat([exposure_sums, sums])
exposure = pd.concat([exposure, zimmermann])


######## read in Meiler et al
meiler = pd.read_excel(input_path + 'dismip_meiler_final.xlsx')
meiler['model_specification'] = meiler['model']
meiler['modeler'] = 'MEI'
meiler = meiler.drop(columns = ['model', 'aad'])


sums_meiler = meiler
sums_meiler['sum_exposed'] = sums_meiler['aae'] * len(years)
sums_meiler = sums_meiler.drop(columns = ['aae'])

## add meiler to aad
aae_df = pd.concat([aae_df, meiler], axis = 0)
sums = pd.concat([sums, sums_meiler], axis = 0)



########### read in riedels data
riedel = pd.read_csv(input_path + 'dismip_riedel_exposure.csv')

riedel= riedel.rename(columns={'Year': 'year', 'Country': 'country', 'People Exposed':'pop_exposed'})
riedel['modeler'] = np.where(riedel['Protection'] == 'No Protection', 'RIE-0', 'RIE-FLO')
riedel = riedel[riedel['year'].isin(range(2008,2022))]
riedel['model_specification'] = riedel['Protection']

riedel = riedel.drop(columns = ['Protection'])


exposure = pd.concat([exposure, riedel], axis = 0)


sums = riedel.groupby(['country','model_specification','modeler']).sum()


sums = sums.reset_index(drop = False)
sums = sums.rename(columns = {'pop_exposed':'sum_exposed'})
sums = sums.drop(columns = ['year'])
sums = sums.drop_duplicates()




aae = riedel.groupby(['country','model_specification','modeler']).mean()
aae = aae.reset_index(drop = False)
aae = aae.rename(columns = {'pop_exposed':'aae'})
aae = aae.drop(columns = 'year')


aae_df = pd.concat([aae_df, aae])
exposure_sums = pd.concat([exposure_sums, sums])



######### read in Kams data
kam = pd.read_excel(input_path + 'dismip_kam_final.xlsx')
kam['modeler'] = 'KAM'
kam['model_specification'] = kam['rcp'] + '+' + kam['model']
kam = kam.drop(columns = ['rcp', 'model', 'pop_displaced'])

sums = kam.groupby(['country','model_specification','modeler']).sum()


sums = sums.reset_index(drop = False)
sums = sums.rename(columns = {'pop_exposed':'sum_exposed'})
sums = sums.drop_duplicates()


aae = kam.groupby(['country','model_specification','modeler']).mean()
aae = aae.reset_index(drop = False)
aae = aae.rename(columns = {'pop_exposed':'aae'})



aae_df = pd.concat([aae_df, aae])
exposure_sums = pd.concat([exposure_sums, sums])
exposure = pd.concat([exposure, kam])


####### read in CIMAS data
cima = pd.read_excel(input_path + 'dismip_cima_exposure.xlsx')
cima['modeler'] = 'CIMA'
cima['model_specification'] = '-'


sums_cima = cima
sums_cima['sum_displacements'] = sums_cima['aae'] * len(years)
sums_cima = sums_cima.drop(columns = ['aae'])


aae_df = pd.concat([aae_df, cima], axis = 0)
print(aad['modeler'].unique())
sums = pd.concat([sums, sums_cima], axis = 0)
print(sums['modeler'].unique())
print(displacements['modeler'].unique())


exposure.to_csv(input_path + 'exposure.csv')
sums.to_csv(input_path + 'sum_exposed.csv')
aae_df.to_csv(input_path + 'aae.csv')


########################################################################################################################
# population
pop = pd.DataFrame()

# riedel
pop_temp = pd.read_csv(input_path + 'dismip_pop_riedel.csv')

pop_temp = pop_temp.melt(id_vars=["Year"], var_name="country", value_name="pop")
pop_temp['modeler'] = 'RIE-0'
pop_temp['SSP'] = 'BLUB'
pop_temp.rename(columns = {'Year':'year'}, inplace = True)

pop = pd.concat([pop, pop_temp])

pop_temp = pd.read_csv(input_path + 'dismip_pop_riedel.csv')
pop_temp = pop_temp.melt(id_vars=["Year"], var_name="country", value_name="pop")
pop_temp['modeler'] = 'RIE-FLO'
pop_temp['SSP'] = 'BLUB'
pop_temp.rename(columns = {'Year':'year'}, inplace = True)

pop = pd.concat([pop, pop_temp])

# kam
pop_temp = pd.read_excel(input_path + 'dismip_pop_kam.xlsx')
pop_temp['modeler'] = 'KAM'
pop = pd.concat([pop, pop_temp])

# zimmermann
pop_temp = pd.read_csv(input_path + 'dismip_zimmermann_pop.csv')
pop_temp['modeler'] = 'ZIM'
pop_temp['SSP'] = 'BLUB'
pop = pd.concat([pop, pop_temp])

# CIMA
pop_temp = pd.read_excel(input_path + 'dismip_cima_pop.xlsx')
pop_temp['year'] = 2010
pop_temp['modeler'] = 'CIMA'
pop_temp['SSP'] = 'BLUB'
pop = pd.concat([pop, pop_temp])

# meiler
pop_temp = pd.read_excel(input_path + 'dismip_meiler_pop.xlsx')
pop_temp['modeler'] = 'MEI'
pop_temp['SSP'] = 'BLUB'
pop_temp['year'] = 2010
pop = pd.concat([pop, pop_temp])

print(pop)

pop.to_csv(input_path + 'pop.csv')



print('Script complete')



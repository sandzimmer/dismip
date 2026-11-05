# Scripts and Data for Manuscript: How (well) can we quantify flood-induced displacement risk? A first intercomparison of models in the Horn of Africa

Reproducible Python scripts for preparing model comparison datasets and producing figures for Ethiopia (ETH), Somalia (SOM), and Sudan (SDN). The original analysis spans 2008–2021, with additional observational or out-of-sample periods in some scripts.

## Repository structure

```text
scripts/                         Python analysis and figure scripts
  merge_model_outputs.py
  process_idmc_gidd_records.py
  compare_idmc_dfo_displacement.py
  evaluate_rie_out_of_sample.py
  plot_average_annual_displacement.py
  plot_aad_robustness.py
  plot_average_annual_exposure.py
  plot_annual_displacement_models.py
  plot_kam_zim1m_annual_displacement.py
  plot_displacement_exposure_ratios.py
  plot_horn_displacement_map.py
dismip/dismip_data/              Model input tables and merged outputs (not included)
dismip/new_idmc_data/            Updated IDMC records and standardized inputs (not included)
dismip/output/                   Generated figures and tables (not included)
data/                            DFO and other observation inputs (not included)
```

All paths are relative to the repository root, determined from the script location. **No user-specific computer or HPC paths are required.** Run scripts from any working directory, provided the repository structure above is maintained.

## Installation

Python 3.10+ recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Some scripts use `geopandas` and geospatial dependencies, which may be easier to install using conda. No original input datasets are included in this archive; obtain them from the respective data providers/collaborators and follow their licensing and sharing restrictions.

## Inputs

| Path relative to repository | Purpose |
|---|---|
| `dismip/dismip_data/aad.csv` | Model average annual displacement |
| `dismip/dismip_data/aae.csv` | Model annual average exposure |
| `dismip/dismip_data/pop.csv` | Model population estimates |
| `dismip/dismip_data/displacements.csv` | Annual modelled displacements |
| `dismip/dismip_data/idps.xlsx` | Conflict-related IDP time series |
| `data/geospatial/gadm36_levels.gpkg` | GADM GeoPackage with `level0` layer |

`merge_model_outputs.py` additionally expects raw contributor-specific CSV/XLSX files (e.g. `dismip_zimmermann_final.csv`, `dismip_meiler_final.xlsx`, `dismip_riedel.csv`, `dismip_kam_final.xlsx`, and `dismip_cima.xlsx`) under `dismip/dismip_data/`. See the script for the complete input list. `process_idmc_gidd_records.py` reads GIDD Excel workbooks under `dismip/new_idmc_data/` and expects the `1_Disaggregated_Data` worksheet. Avoid placing unrelated XLSX files in that directory when running it.

Other scripts may require additional inputs: inspect their input constants at the top of each file. The input schemas and scientific assumptions have **not** been standardized across all scripts.

## Suggested workflow

1. Populate the `data/`, `dismip/dismip_data/`, and `dismip/new_idmc_data/` directories with the required input files.
2. Run `python scripts/process_idmc_gidd_records.py` only if you need to regenerate IDMC aggregates from raw GIDD workbooks. Review its event-filtering and deduplication assumptions before using outputs.
3. Run `python scripts/merge_model_outputs.py` if you need to regenerate the merged model tables. It reads collaborator inputs and ZIM-D predictions and writes merged tables under `dismip/dismip_data/`.
4. Run individual plotting scripts, for example:

```bash
python scripts/plot_average_annual_displacement.py
python scripts/plot_average_annual_exposure.py
python scripts/plot_displacement_exposure_ratios.py
python scripts/plot_annual_displacement_models.py
python scripts/plot_horn_displacement_map.py
```

Figures and intermediate results are generally saved in `dismip/output/`. Some preprocessing scripts write into the input-data directories, retaining their original behavior.

## Script guide

| Script | Function |
|---|---|
| `merge_model_outputs.py` | Assemble contributor model data into annual displacement, AAD, exposure, AAE and population tables |
| `process_idmc_gidd_records.py` | Explore/classify GIDD flood records and create updated IDMC summaries |
| `compare_idmc_dfo_displacement.py` | Compare IDMC and DFO annual country displacement records |
| `evaluate_rie_out_of_sample.py` | Compare RIE estimates with IDMC for in- and out-of-sample periods |
| `plot_average_annual_displacement.py` | Grouped event-based/probabilistic AAD comparison |
| `plot_aad_robustness.py` | AAD comparison with alternative IDMC benchmark periods |
| `plot_average_annual_exposure.py` | AAE and population, logarithmic and linear versions |
| `plot_annual_displacement_models.py` | ZIM-D verified ensemble and annual time-series comparison |
| `plot_kam_zim1m_annual_displacement.py` | Annual KAM and ZIM-1M comparison |
| `plot_displacement_exposure_ratios.py` | Modelled/IDMC/DFO displacement as fraction of model exposure |
| `plot_horn_displacement_map.py` | Horn of Africa overview map with displacement/IDP time series |

## Reproducibility notes

- Scripts were **path-cleaned and syntax-checked**, but could not be executed end-to-end because the original datasets and geospatial files were not supplied.
- Scientific calculations, filters, figures and model-specific uncertainty handling were intentionally retained. **The scripts are not guaranteed to be mutually consistent** about IDMC columns, DFO allocation methods, or observational benchmark periods; check these against the manuscript before publication.
- The GADM input referenced in the original map script is **GADM 3.6**, despite possible references elsewhere to GADM 4.1. To use GADM 4.1, confirm layer names, ISO identifiers and boundaries before switching files.
- The GIDD-processing script is exploratory and contains multiple sequential analyses; inspect its intermediate outputs and filtering assumptions before treating it as a production pipeline.
- Check licensing for GADM, IDMC, DFO, and contributor datasets before distributing them. Only code is included here.


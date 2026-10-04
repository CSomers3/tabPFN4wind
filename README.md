# windpfn: day-ahead wind forecasts for Great Britain with TabPFN-3.5

Wind generated a third of Great Britain's metered electricity in 2025, and the grid is scheduled a day ahead on the system operator's forecast of it. What that forecast misses is made good in real time: NESO's WINDFOR missed by 9.4 TWh in 2025, a year in which balancing the grid cost £2.3bn. Each tenth of that miss costs about £20m a year, and the cost grows with every gigawatt of wind built.

windpfn passes NESO's forecast, public ECMWF weather and every settled hour of outturn to TabPFN-3.5, a tabular foundation model, with no training or tuning. Over 623 held-out days it cuts NESO's miss by 11–14% and returns a calibrated probability distribution for every hour. It runs [live](https://csomers3.github.io/tabPFN4wind/), fifteen minutes after each NESO update.

![Cumulative absolute error over 14,952 held-out hours, 2025–26: NESO's forecast reaches 15.3 TWh, TabPFN-3.5 post-processing it on AIFS weather reaches 13.1 TWh, 14.3% less](notebooks/miss.png)

## Results

The test covers 2025-01-01 → 2026-09-15 and was scored under a protocol frozen beforehand ([`8e4386d`](../../commit/8e4386d)). The frozen run read ECMWF IFS weather. The pipeline then moved to the open AIFS archive and was rerun with nothing else changed. Both runs are shown.

| 14,952 hours, MW | MAE | ΔMAE vs NESO [95% CI] | CRPS | 10–90% coverage | Winkler |
|---|---|---|---|---|---|
| NESO WINDFOR | 1,022 | | 1,022 | | |
| WINDFOR + conformal band ¹ | 1,004 | −18 [−30, −6] | 788 | 79.2% | 4,659 |
| LightGBM · AIFS ¹ | 929 | −93 [−121, −65] | 737 | 67.2% | 4,492 |
| **TabPFN-3.5 · IFS** ² | **912** | **−109 [−146, −72]** | **714** | **80.6%** | **4,195** |
| **TabPFN-3.5 · AIFS** ³ | **876** | **−146 [−183, −110]** | **685** | **81.9%** | **4,009** |

Lower is better, and coverage targets 80%. A point forecast's CRPS is its MAE. The intervals are 95% day-block bootstrap.
¹ Added after the test and tuned on 2024 only. ² The frozen one-shot test, on ECMWF IFS at 100 m. ³ The same protocol on ECMWF AIFS at 10 m, which the package and the live forecast read.

- **The gain needs weather, and TabPFN uses it best.** Recalibrating WINDFOR by its own errors recovers 18 MW, and TabPFN on weather alone only ties NESO. On identical AIFS inputs, TabPFN beats a tuned LightGBM by 53 MW (95% CI 24 to 84).
- **Its distribution is calibrated without a calibration step.** Every central interval covers within 2.5 points of nominal, and its p10–p90 is 11% narrower than the conformal band. LightGBM's quantiles fall up to 13 points short.
- **It fails when the weather does.** On the eight days it lost worst to NESO, outturn fell inside its p10–p90 in only 35–42% of hours. During Storm Éowyn it overshot by 2.2–2.3 GW on the day, while NESO was within 0.1 GW.

![Left: coverage minus nominal from p40–p60 to p10–p90, both TabPFN-3.5 runs within 2.5 points above zero, NESO + conformal within 1 point below, LightGBM 5–13 points short. Right: mean interval width, TabPFN-3.5 narrower than NESO + conformal at every level](notebooks/calibration.png)

Prior Labs' [performance guidance](https://docs.priorlabs.ai/improving-performance) was tested on 2024 data ([`02`](notebooks/02_development.py)). Feature ratios, native datetime handling, Thinking mode and per-estimator subsampling all failed to beat the defaults.

## The model

From [src/windpfn/forecasting.py](src/windpfn/forecasting.py):

```python
model = TabPFNRegressor.create_default_for_version("v3.5")
model.fit(features[train], (table.y / table.cap)[train])
deciles = model.predict(features[test], output_type="quantiles", quantiles=QUANTILES)
```

`features` holds 66 numeric columns: power-curve capacity factor and wind direction at 20 wind-farm clusters, capacity, lead time, calendar and WINDFOR. `train` is every hour settled a day before issue, so each monthly refit only lengthens the context: 6,944 hours in January 2025, 21,537 by September 2026. With the context frozen at January 2025, TabPFN still scores 922 MW on IFS and 886 MW on AIFS.

## Live

A [GitHub Action](.github/workflows/live.yml) runs after each of NESO's eight daily updates. It issues TabPFN-3.5's p1–p99 for every hour the update covers and commits them to [`notebooks/public/live.csv`](notebooks/public/live.csv), so the git log holds each forecast before its outturn exists. The [live page](https://csomers3.github.io/tabPFN4wind/) draws outturn, fetched from Elexon on load, under both forecasts. For past hours it shows the last forecast issued before each.

![The live page: outturn since yesterday under TabPFN-3.5's median inside its p10–p90, p20–p80, p30–p70 and p40–p60 fan and NESO's forecast, each from the last update before the hour, to the end of tomorrow](notebooks/live.png)

## Reproduce

Python 3.12 or later.

```bash
git clone https://github.com/CSomers3/tabPFN4wind && cd tabPFN4wind
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[tabpfn,notebooks]"

windpfn-score     # the results table, offline, from the committed forecasts
windpfn-sample    # the whole pipeline on a bundled six-month sample, offline, in about a minute
```

TabPFN-3.5 runs on Prior Labs' hosted API. To include it, set a token from a Prior Labs account: `export TABPFN_TOKEN=<token>` (PowerShell: `$env:TABPFN_TOKEN = "<token>"`), then run `windpfn-sample --tabpfn`. The sample's context is months, not years, so its scores are a smoke test, not a result.

| | |
|---|---|
| `marimo edit notebooks/sample.py` | `windpfn-sample` as a notebook: inputs, forecasts, scores and any week's chart |
| `marimo edit notebooks/03_test.py` | held-out results and figures, offline |
| `marimo edit notebooks/02_development.py` | the 2024 development runs and the Prior Labs configuration study (token) |
| `marimo run notebooks/explorer.py` | the live page, locally |
| `windpfn-live` | issue p1–p99 from NESO's newest update (token) |
| `windpfn-backtest tabpfn` | regenerate TabPFN's held-out forecasts (raw pulls and a token) |
| `windpfn-backtest references` | regenerate LightGBM's and the conformal band's |
| `windpfn-fetch` | pull every raw input into `data/raw/`, each logged with its URL and sha256 |

The pipeline is five modules in [src/windpfn/](src/windpfn): `sources`, `weather`, `dataset`, `forecasting` and `evaluation`. The notebooks only read from it.

## Data

| | |
|---|---|
| `data/history/` | every input from 2024-04-01 to 2026-09-30, from Elexon BMRS (WINDFOR vintages, metered wind, wind curtailment, capacity) and ECMWF AIFS via dynamical.org |
| `data/sample/` | the same for 2026-03-01 → 2026-09-15 |
| `data/points.csv` | the 20 weather points: capacity-weighted clusters of REPD wind farms |
| `data/test_forecasts.parquet` | TabPFN-3.5's held-out forecasts on IFS, the frozen test |
| `data/test_forecasts_aifs.parquet` | the same on AIFS |
| `data/reference_forecasts.parquet` | LightGBM's and the conformal band's |

## Limitations

- **This is post-processing.** It needs NESO's forecast as an input.
- **The target is reconstructed.** NESO publishes no unconstrained outturn, so the target is metered wind plus balancing-mechanism curtailment. Self-curtailment at negative prices is missing, in about 3% of hours.
- **The weather input is coarse**: 20 points, 0.25°, 6-hourly steps, and 10 m wind on AIFS.
- **The AIFS run came after the IFS score was known**, with no other change.

Data: Elexon BMRS, ECMWF AIFS via dynamical.org (CC BY 4.0), REPD (OGL v3.0). Code: MIT.

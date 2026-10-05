![Line drawing of an offshore wind farm; cartoon wind curls sweep across and the turbines speed up as each one passes](.github/hero.svg)

# TabPFN-3.5 for Great Britain's day-ahead wind forecast

TabPFN-3.5 post-processes the system operator's day-ahead wind forecast from public weather and outturn data, with no training and no tuning.
On 623 held-out days, the median forecast has 11% lower mean absolute error than NESO's (912 vs 1,022 MW), and the 80% interval holds 80.6% of hours.
Live since October 2026, after every NESO update: **[csomers3.github.io/tabPFN4wind](https://csomers3.github.io/tabPFN4wind/)**

Wind supplied a third of Great Britain's electricity in 2025, and the grid is scheduled a day ahead on NESO's forecast of it. That forecast missed by 9.4 TWh in 2025, a year in which balancing the grid cost £2.3bn. Each 10% cut in the miss is worth about £20m a year at the gap between imbalance and market prices.

## Results

Held out: 2025-01-01 to 2026-09-15, 14,952 hours, scored once under the protocol frozen at commit [`8e4386d`](../../commit/8e4386d). All intervals are central intervals of the predictive distribution.

| MW | MAE | ΔMAE vs NESO [95% CI] | CRPS | 80% interval coverage | Winkler |
|---|---|---|---|---|---|
| NESO | 1,022 | | 1,022 | | |
| NESO recalibrated + conformal interval | 1,004 | −18 [−30, −6] | 788 | 79.2% | 4,659 |
| **TabPFN-3.5** | **912** | **−109 [−146, −72]** | **714** | **80.6%** | **4,195** |
| TabPFN-3.5 · AIFS | 876 | −146 [−183, −110] | 685 | 81.9% | 4,009 |

TabPFN-3.5 read ECMWF IFS weather in the pre-registered test; the AIFS rerun reproduces the gain on open data. The conformal reference was added after the test and tuned on 2024 only: its median is NESO's forecast shifted by NESO's own error quantiles within five forecast-level bins, so its MAE differs from NESO's. CRPS is twice the mean pinball loss over nine deciles, and equals MAE for a point forecast. Winkler is the 80% interval's width plus 10 times any distance outside it. Confidence intervals resample whole days.

## Why TabPFN

**Calibration.** TabPFN-3.5's central intervals are within 2.5 points of nominal coverage at every level, with no calibration step, and its 80% interval is 11% narrower than the conformal one.

![Left: coverage minus nominal, for intervals of nominal coverage 20% to 80%. Right: mean interval width.](notebooks/calibration.png)

*Coverage minus nominal, for intervals of nominal coverage 20% to 80% (left), and mean interval width (right).*

**Against tuned references.** A quantile LightGBM and a quantile regression forest were given TabPFN-3.5's exact AIFS inputs, context and monthly refits, with hyperparameters chosen by CRPS on Jul–Dec 2024 ([`04`](notebooks/04_references.py)):

| MW | MAE | ΔMAE vs TabPFN-3.5 [95% CI] | CRPS | 80% interval coverage | Winkler |
|---|---|---|---|---|---|
| **TabPFN-3.5 · AIFS** | **876** | | **685** | **81.9%** | **4,009** |
| LightGBM | 891 | +16 [−4, +36] | 706 | 66.2% | 4,325 |
| Quantile regression forest | 901 | +25 [+8, +44] | 713 | 88.0% | 4,279 |

The medians are close; the distributions are not. LightGBM's 80% interval holds 66% of hours and the forest's 88%. With the context cut to 250 rows a month, TabPFN-3.5 still beats NESO, at 921 MW; LightGBM falls to 1,148 and the forest to 1,051.

![Held-out MAE against context rows per monthly fit, for TabPFN-3.5, the two references and NESO.](notebooks/scaling.png)

**Prior Labs' performance suggestions did not beat the defaults.** Each was scored on Jul–Dec 2024, where the default setup has an MAE of 805 MW against NESO's 931 ([`02`](notebooks/02_development.py)):

| Suggestion | Jul–Dec 2024 MAE, MW |
|---|---|
| Default TabPFN-3.5, 66 numeric columns | 805 |
| Ratio features: NESO's forecast as capacity factor; a fleet-wide power curve | 809; 806 |
| Native datetime column in place of hour and day-of-year sin/cos | 818 |
| Thinking mode (mean only, no distribution), against the default mean | 799 vs 809, CI includes 0, about 5× the tokens |
| Per-estimator subsampling (50,000 rows each) against one 50,000-row draw, every NESO update, Nov–Dec | 868 vs 864 |

## How it works

Each forecast used only what was public at its issue time, 15 minutes after NESO's last update before 08:30 UTC on the previous day (lead times 16 to 39 h). `dataset.check` enforces the timing on every row.

- **Weather**: ECMWF 6-hourly 10 m wind at 20 capacity-weighted points, one per cluster of wind farms in the Renewable Energy Planning Database, raised to hub height by the one-seventh power law and mapped through a generic power curve. Wind direction entered as sin and cos.
- **NESO's forecast**: the update being post-processed.
- **Target**: metered transmission wind plus balancing-mechanism curtailment of wind units, divided by installed capacity. This curtailment adjustment reconstructs what NESO forecasts: regressed on NESO's forecast, the slope is 1.00 against 0.80 for metered output alone.
- **Context**: every hour settled at least a day before issue, refit monthly; 6,944 hours in January 2025, 21,537 by September 2026.

```python
model = TabPFNRegressor.create_default_for_version("v3.5")
model.fit(features[train], (table.y / table.cap)[train])
deciles = model.predict(features[test], output_type="quantiles", quantiles=QUANTILES)
```

The point forecast is the median. The live job reads 99 percentiles from the full predictive distribution.

**Design decisions.** Features, target, context and embargo were fixed on 2024 data before the test. Capacity factor, not MW, was the target, so the context spans capacity growth. A one-day embargo covers curtailment data whose first publication time cannot be shown. Rejected: weather-only forecasts (TabPFN-3.5 ties NESO at 1,026 MW) and the Prior Labs suggestions above.

## Reproduce

```bash
make score    # Docker: builds the package and prints every table above from the committed forecasts
```

Without Docker, on Python 3.12:

```bash
git clone https://github.com/CSomers3/tabPFN4wind && cd tabPFN4wind
pip install -e ".[tabpfn,references,notebooks]"
windpfn-score                     # the tables above, offline, in seconds
marimo edit notebooks/sample.py   # the whole pipeline on a six-month sample
```

| Command | Output |
|---|---|
| `windpfn-fetch` | every raw input from Elexon BMRS, dynamical.org and NESO, each logged with its URL and sha256 checksum |
| `windpfn-backtest tabpfn` | the held-out TabPFN-3.5 forecasts from `data/history/` (needs `TABPFN_TOKEN`) |
| `windpfn-backtest references`, `ablation`, `context` | the reference forecasts, the ablation on identical inputs, and the context-size runs |
| `windpfn-live` | a new live forecast |

`data/` holds every input from April 2024, the six-month sample and all committed held-out forecasts. Notebooks [`01`](notebooks/01_data.py) to [`05`](notebooks/05_live.py) read only from the package in `src/windpfn/` and rebuild every number and figure; [`docs/method.md`](docs/method.md) is the method note. The [scheduled GitHub Action](https://github.com/CSomers3/tabPFN4wind/actions/workflows/live.yml) runs after each of NESO's eight daily updates and commits each forecast to `notebooks/public/live.csv` before its outturn exists.

## Limitations

- The method post-processes NESO's forecast and needs it as an input.
- The target is reconstructed. Wind farms that curtail themselves at negative prices, in about 3% of hours, are missing from it.
- The weather input is coarse: 20 points at 0.25°, 6-hourly, 10 m wind.
- Errors follow the weather forecast. On the eight days TabPFN-3.5 lost most to NESO, 35% to 42% of hours fell inside its 80% interval. During Storm Éowyn (24 January 2025) its daily bias was +2.2 GW against NESO's −0.1 GW.
- The AIFS rerun and the references followed the test score; no TabPFN-3.5 setting changed.

Data: contains BMRS data © Elexon Limited copyright and database right 2026; NESO balancing costs, supported by National Energy System Operator Open Data; ECMWF AIFS via dynamical.org (CC BY 4.0); REPD (OGL v3.0); Natural Earth (public domain). Code: [Apache-2.0](LICENSE).

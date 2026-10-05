# Forecasting Great Britain's wind a day ahead with TabPFN-3.5

*Method note for [windpfn](../README.md). Every number here is rebuilt by the notebook named beside it.*

## Problem

Wind generated a third of Great Britain's metered electricity in 2025, and Britain plans to more than double its wind capacity by 2030. The grid is scheduled a day ahead on the system operator's forecast of that wind, NESO's WINDFOR. What WINDFOR misses has to be made good in real time, by calling on reserve plant or by turning generation down. In 2025 it missed by 9.4 TWh, and NESO spent £2.3bn balancing the system. Priced at the gap between imbalance and market prices, each tenth of that miss costs about £20m a year ([`01`](../notebooks/01_data.py)). That cost grows with every gigawatt of wind built.

We ask whether a tabular foundation model can improve that forecast from public data alone, with no training, and say how far tomorrow could miss by.

## Information set

Each delivery day D is forecast from what was public at the issue time, fifteen minutes after the last WINDFOR update published by D−1 08:30 UTC. Lead times are 16–39 h. `dataset.check` enforces the timing on every row.

- **NESO's forecast**: the WINDFOR vintage itself.
- **Weather**: the newest ECMWF run public at issue. The model reads wind speed and direction at 20 points, each the capacity-weighted centre of a cluster of wind farms from the Renewable Energy Planning Database (REPD). Each speed is mapped to capacity factor through a generic turbine power curve.
- **Context**: every hour whose outturn had settled a day before issue. The day's embargo covers curtailment data whose original publication time cannot be shown. The context expands and the model is refit monthly.
- **Target**: available wind, meaning metered transmission wind (FUELHH) plus balancing-mechanism curtailment of wind units (BOAV), hour-ending, divided by installed capacity (B1610). NESO publishes no unconstrained outturn, so this reconstructs what WINDFOR forecasts. Outturn regressed on WINDFOR has slope 1.00 on this target, against 0.80 on metered output alone.

## Model

TabPFN-3.5 on Prior Labs' hosted API, with default settings. It takes 66 columns: the 20 power-curve capacity factors, sin and cos of the 20 wind directions, capacity, lead time, hour, day-of-year sin and cos, and WINDFOR. It returns a predictive distribution over capacity factor. The point forecast is its median, the MAE-optimal summary. The held-out scores read nine deciles, and the live forecast reads p1–p99.

## Evaluation

The protocol was [frozen at `8e4386d`](../../../blob/8e4386d/docs/method.md) on 2024 data alone, before any test score existed. The held-out test runs from 2025-01-01 to 2026-09-15: 623 days and 14,952 hours, scored once ([`03`](../notebooks/03_test.py)).

- **MAE**, and ΔMAE against WINDFOR with a 95% day-block bootstrap interval. Hourly errors are strongly autocorrelated (0.89 at one hour).
- **CRPS**, approximated as twice the mean pinball loss over the nine deciles.
- **The 10–90% interval**: its coverage, its width, and the Winkler score, which adds a penalty for each miss to the width.

The frozen test read ECMWF IFS open data at 100 m. The package then moved to ECMWF's AIFS archive on dynamical.org, which anyone can query by run. It serves 10 m wind for the whole period, raised to hub height by the one-seventh power law. TabPFN was rerun on AIFS with nothing else changed. We report both runs. The references were added after the test and tuned on 2024 only: NESO's forecast recalibrated with a conformal interval, and a quantile LightGBM and a quantile regression forest given TabPFN's exact AIFS inputs, context and refits ([`04`](../notebooks/04_references.py)).

## Results

| 14,952 held-out hours, MW | MAE | ΔMAE vs NESO [95% CI] | CRPS | 80% interval coverage | Winkler |
|---|---|---|---|---|---|
| NESO | 1,022 | | 1,022 | | |
| NESO recalibrated + conformal interval | 1,004 | −18 [−30, −6] | 788 | 79.2% | 4,659 |
| LightGBM · AIFS | 891 | −130 [−160, −102] | 706 | 66.2% | 4,325 |
| Quantile regression forest · AIFS | 901 | −120 [−150, −92] | 713 | 88.0% | 4,279 |
| **TabPFN-3.5 · IFS** | **912** | **−109 [−146, −72]** | **714** | **80.6%** | **4,195** |
| **TabPFN-3.5 · AIFS** | **876** | **−146 [−183, −110]** | **685** | **81.9%** | **4,009** |

1. **The gain needs weather, and TabPFN uses it best.** Recalibrating WINDFOR by its own errors recovers 18 MW. TabPFN on weather alone only ties NESO (1,026 and 1,032 MW). Given both, it cuts NESO's miss by 11–14%. On identical AIFS inputs its median beats a tuned quantile LightGBM by 16 MW (95% CI −4 to 36), within noise, and a quantile regression forest by 25 MW (8 to 44). With 250 context rows a month the gaps widen to 227 and 130 MW.
2. **Its distribution is calibrated without a calibration step.** At every central interval from p40–p60 to p10–p90, coverage is within 2.5 points of nominal, and its p10–p90 is 11% narrower than the conformal band. On the same inputs LightGBM's p10–p90 holds 66% of hours and the forest's 88%.
3. **It fails when the weather does.** On the eight days it lost worst to NESO, outturn fell inside its p10–p90 in only 35–42% of hours. On 24 January 2025, during Storm Éowyn, NESO's daily bias was −0.1 GW and TabPFN's was +2.2 to +2.3 GW. Calibrated on average is not calibrated on the days that matter most.

## What Prior Labs' guidance changed

We tested each lever in [Improving performance](https://docs.priorlabs.ai/improving-performance) that applies here. Each change was scored against the setup above on Jul–Dec 2024, inside the development window ([`02`](../notebooks/02_development.py)). That setup scores 805 MW there, against WINDFOR's 931.

| Lever | Here | Jul–Dec 2024 |
|---|---|---|
| Raw columns, declared types | All 66 columns are numeric, with no categories, text or gaps | nothing to declare |
| Ratio and aggregate features | WINDFOR as capacity factor; a fleet-wide power curve | 809 and 806 MW, within noise |
| Native datetime handling | Valid time as a datetime column in place of the hand-made calendar | 818 MW, no better |
| More estimators | The hosted API caps `n_estimators` at 8, the default | not available |
| Thinking mode, with `time_col` | Its regression returns only the mean, so no distribution | 799 MW against 809 for the standard mean (CI includes zero), at about 5× the tokens |
| Per-estimator row subsampling | Live context of all ~400,000 vintage rows, against one 50,000-row draw | 868 against 864 MW (Nov–Dec 2024), so the single draw is kept |
| KV cache | Each fit predicts once | no benefit |
| Temperature and metric tuning | `tuning_config` is local-only, and coverage is already on target | not needed |

None of the levers improved on the defaults. For a narrow, numeric, well-specified table, the default TabPFN-3.5 was already the right model, and the remaining error lies in the weather input.

## Limitations

- **This is post-processing.** It needs WINDFOR as an input, and on weather alone TabPFN only ties NESO.
- **The target is reconstructed.** Self-curtailment at negative prices is not a balancing action and is missing from it, in about 3% of hours.
- **The weather input is coarse**: 20 points, at 0.25°, in 6-hourly steps. The AIFS run uses 10 m wind because the archive's 100 m wind starts only in February 2025.
- **The AIFS run came after the IFS score was known.** No setting changed between them; the weather source changed so that others can run the pipeline.

## Next

Ensemble weather spread as an input is the most direct fix for the storm-day failure: the model should widen its band when the weather models disagree. So is 100 m wind, once the archive covers the whole context. The [live forecast](https://csomers3.github.io/tabPFN4wind/) is the forward test. Each forecast is committed to git before its outturn exists.

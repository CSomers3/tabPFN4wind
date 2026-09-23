# Phase 0 — data audit (2026-09-21)

Figures: `00_audit.ipynb`. Raw pulls are cached in `data/raw/`, with URLs and sha256 in `manifest.jsonl`.

## WINDFOR (NESO wind forecast)
- `/datasets/WINDFOR/stream` accepts 1-month publish-time windows. Values are hourly.
- Vintages from 2016-03. 4/day until 2019-09, 8/day after. 49 rows per vintage until 2024-10, 73 after.
- Stamps: winter 02:30/04:30/07:30/09:30/11:30/15:30/18:30/22:30Z; summer +1 h. **There is no 08:30Z vintage in winter.** Winter timing is not yet verified live (`scripts/poll_windfor_latency.py`).
- Each vintage repeats elapsed hours unchanged, so drop `startTime <= publishTime`.
- The latest vintage ≤ D-1 08:30Z covers all 24 UTC hours of D on 100% of days 2022–26, and 97–99% before that.
- Timestamp semantics are ambiguous: MAE differs by <3% across hour-start, centred and hour-ending alignments against 5-min FUELINST (2021).
- DGWS (day-ahead wind and solar) is issued at 16:45Z D-1, after our cutoff, and has a different scope. It is not a usable benchmark.

## Outturn (FUELHH WIND, 30-min means)
- 2016 → now, ~130 missing half-hours in total.
- **The API's (settlementDate, SP) labels are wrong** until 2022-07-14 (SP48 and clock-change SPs carry the next day's date). Always key on UTC `startTime`; keyed that way, clock-change days have 46/50 periods.
- WINDFOR − FUELHH bias by WINDFOR decile (2019–21, MW): `17 -43 -117 -15 -35 156 195 349 638 1555`. It is concentrated at high output, which is the curtailment signature. See `01_eda` §4.1.

## NWP
Open-Meteo was rejected. Its Historical Forecast API is a near-analysis, Previous Runs `_day1` leaks, and Single Runs has no publication times. We use ECMWF IFS open data instead: 0.25° from 2024-03, with publication time taken from mirror `Last-Modified`.

## Capacity
- `/reference/bmunits/all` is a current snapshot only (285 wind units, 30.8 GW, no dates).
- Trailing max outturn plateaus at 17–18 GW over 2023–26 because constraints cap it, so it is not usable as a capacity proxy.
- Used instead: B1610 per-unit first metered output (+7 d publication lag) × snapshot nameplate.

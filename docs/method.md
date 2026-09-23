# Evaluation protocol — frozen 2026-09-23

Frozen by the git commit that adds `03_test.ipynb`, before any 2025+ score was computed.

## Information set (per delivery day D, UTC)
- `τ`: the latest WINDFOR publishTime ≤ D-1 08:30Z. In practice that is 07:30Z in GMT and 08:30Z in BST.
- Issue time: τ + 15 min, since the vintage becomes visible ~2 min after its publishTime. NWP and outturn cutoffs stay at τ or earlier.
- NWP: ECMWF IFS 0.25° open data (100 m u/v, 3-hourly, nearest grid point to 20 REPD capacity clusters). Use the newest run whose earliest upload across the GCS and AWS mirrors (`Last-Modified`) is ≤ τ; fall back to older runs if a file is missing on GCS (25 days of 2024 use a 12Z run for this reason). In 2024 that was 00Z D-1 on 72% of days; in 2025, on ~7%.
- Capacity: B1610 wind BMUs (T_, E_) with a weekly sample showing output at or before τ − 7 d, × snapshot nameplate.
- Model features: see Model. Recent outturn is not used (leads are 16–39 h; on 2024 it made the model worse).
- Enforced by `dataset.check`, and tested in `tests/test_timing.py`.

## Target
Available transmission wind: FUELHH wind + curtailment, where curtailment = −2 × Σ BOAV accepted volumes (bids net of offers, MWh per half-hour) over the FUELHH wind BMUs. Hour-ending mean over [t−1h, t).
Rationale (2024 exploration window, `01_eda` §4.1): outturn regressed on WINDFOR has slope 1.00 on this target vs 0.80 on FUELHH alone. Regressing WINDFOR on both parts gives 0.95·metered + 0.90·curtailed, so WINDFOR does not net out curtailment. Raw WINDFOR is therefore the like-for-like benchmark. Metered FUELHH is reported as a secondary target.
Fidelity: NESO publishes no "unconstrained outturn", so WINDFOR's exact target cannot be observed; `y` reconstructs it. Averaged by forecast level, WINDFOR sits a near-constant 150–300 MW above `y` at positive prices, which points to outages/derates or a small unit-scope difference. Removing the whole offset would cut WINDFOR's MAE by only 2% (951 → 932).
Known gap: self-curtailment at negative prices (via physical notifications) is not a BM action and is missing from `y`. WINDFOR − `y` is ~+900 MW below −£10/MWh (3% of hours).
In backtests the target is never a feature. BOAV is published ~1 h after each half-hour live, but archived rows carry createdDateTime ≈ startTime + 25 h, so their historical availability cannot be shown.

## Model (`windpfn/model.py`)
- TabPFN-3.5 (`tabpfn-client`, `v3.5` default checkpoint), predictive median (the full deciles are saved for diagnostics). Target `y / cap`, rescaled by `cap`.
- Features (`model.features`): power-curve capacity factor and sin/cos of 100 m wind direction at the 20 points, `cap`, `lead_h`, `hour`, `doy_sin`, `doy_cos`. Claim (b) adds `windfor`.
- Context: every hour with valid_time ≤ (first issue of the month − 1 day), expanding, refit monthly. The 1-day embargo covers the unverifiable BOAV timing. The `frozen` variant keeps the context of the first test month throughout, isolating the value of refitting.

## Claims (reported separately)
(a) TabPFN, weather only, vs WINDFOR. Development (Apr–Dec 2024): MAE 931 vs 931, ΔMAE 0 [−60, +57].
(b) The same plus WINDFOR as a feature (post-processing). Development: MAE 818, ΔMAE −113 [−161, −64]. The bar is the fixed 50/50 WINDFOR/power-curve blend (811 on Aug–Dec 2024), not WINDFOR.

## Scoring
- All hours, target `y`. Regime splits only by quantities known at issue (forecast level, lead, run age, season), **never by realised curtailment**: that selects on the outcome.
- ΔMAE vs WINDFOR with a 95% day-block bootstrap interval (`model.score`; hourly error ACF 0.89 at 1 h).
- Primary: monthly-refit runs. Frozen runs reported alongside. Also by month and by lead.

## Windows
- NWP archive start 2024-03-16. Exploration and tuning use 2024-03-16 → 2024-12-31 only.
- Test: 2025-01-01 → 2026-09-15, scored once (`03_test.ipynb`).
- Live forward test from launch.

## Disclosures
- 2026-09-22: a one-week smoke test (2025-01-14..20) printed the correlations of the raw physical baseline and WINDFOR with outturn (both 0.957). No model was fitted or tuned on it.
- 2026-09-22/23: features, target scaling, context and embargo were chosen on 2024 only (`02_tabpfn.ipynb`).

## Open decisions
ACI step size, the Dogger Bank A/B coordinates in REPD, and whether to build the 20 points from metered farms only (REPD includes ~4.6 GW of embedded wind that FUELHH never sees).

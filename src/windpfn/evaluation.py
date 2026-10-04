"""Scoring, and the committed held-out forecasts it runs on. Everything here is in MW.

The held-out window was scored once, under the protocol frozen at FREEZE_COMMIT, with ECMWF IFS
weather. TabPFN was then rerun unchanged on the AIFS archive the package reads. Both sets of
forecasts are committed in data/, so the scorecard rebuilds offline without raw pulls or a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from windpfn import forecasting, sources

FREEZE_COMMIT = "8e4386d"
RESULTS_COMMIT = "d7df551"
TEST_START, TEST_END = "2025-01-01", "2026-09-15"

WEATHER = {"ifs": "test_forecasts.parquet", "aifs": "test_forecasts_aifs.parquet"}
PROBABILISTIC = [
    "conformal",
    "lgbm",
    "lgbm+windfor",
    *[f"{run} {weather}" for run in ("tabpfn", "tabpfn+windfor", "tabpfn+windfor frozen") for weather in WEATHER],
]

BOOTSTRAP_DRAWS = 2000
ALPHA = 0.2  # miscoverage of the 10-90% interval


def winkler_score(y, lower, upper, alpha: float = ALPHA) -> pd.Series:
    """Width plus 2/alpha times the distance outside. Lower is better."""
    outside = (lower - y).clip(lower=0) + (y - upper).clip(lower=0)
    return upper - lower + 2 / alpha * outside


def point_scores(errors: pd.DataFrame, reference: str = "windfor", seed: int = 0) -> pd.DataFrame:
    """MAE, RMSE and bias per column, and ΔMAE vs `reference` with a 95% day-block bootstrap interval."""
    daily = errors.abs().groupby(errors.index.floor("D")).mean()
    gain = daily.sub(daily[reference], axis=0)
    draws = np.random.default_rng(seed).integers(0, len(daily), (BOOTSTRAP_DRAWS, len(daily)))
    boot = gain.to_numpy()[draws].mean(1)
    return pd.DataFrame(
        {
            "MAE": errors.abs().mean(),
            "RMSE": errors.pow(2).mean() ** 0.5,
            "bias": errors.mean(),
            "ΔMAE": gain.mean(),
            "lo": np.percentile(boot, 2.5, 0),
            "hi": np.percentile(boot, 97.5, 0),
        }
    ).round(0)


def probabilistic_scores(deciles: dict[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
    """CRPS, and width, coverage and Winkler score of the 10-90% interval.

    CRPS is approximated as twice the mean pinball loss over the nine deciles.
    """
    levels = np.array(forecasting.QUANTILES)

    def one(forecast):
        error = y.to_numpy()[:, None] - forecast.to_numpy()
        pinball = np.maximum(levels * error, (levels - 1) * error).mean()
        return {
            "CRPS": 2 * pinball,
            "width80": (forecast.q90 - forecast.q10).mean(),
            "cover80": y.between(forecast.q10, forecast.q90).mean(),
            "winkler80": winkler_score(y, forecast.q10, forecast.q90).mean(),
        }

    return pd.DataFrame({name: one(forecast) for name, forecast in deciles.items()}).T


def load_forecasts() -> pd.DataFrame:
    """The committed held-out forecasts with outturn `y`, WINDFOR and the blend, per hour. TabPFN's
    runs are tagged with their weather, as 'tabpfn+windfor ifs' and 'tabpfn+windfor aifs'."""
    frames = {weather: pd.read_parquet(sources.DATA / name) for weather, name in WEATHER.items()}
    tabpfn = {f"{run} {weather}": deciles for weather, frame in frames.items() for run, deciles in forecasting.unpack(frame).items()}
    columns = frames["ifs"].drop(columns=frames["ifs"].filter(like="|").columns)
    references = pd.read_parquet(sources.DATA / "reference_forecasts.parquet")
    return columns.join(forecasting.pack(tabpfn)).join(references)


def load_lead_forecasts() -> pd.DataFrame:
    """The committed held-out forecasts from every WINDFOR vintage, one row per (valid_time, issue_time)."""
    return pd.read_parquet(sources.DATA / "lead_forecasts.parquet")


def scorecard(forecasts: pd.DataFrame, names=PROBABILISTIC):
    """Deciles per model, point errors, and the joint table. A point forecast's CRPS is its MAE."""
    runs = forecasting.unpack(forecasts)
    deciles = {name: runs[name] for name in names}
    medians = {name: forecast.q50 for name, forecast in deciles.items()}
    errors = forecasts.filter(["windfor", "blend"]).join(pd.DataFrame(medians)).sub(forecasts.y, axis=0)
    table = point_scores(errors).join(probabilistic_scores(deciles, forecasts.y).astype(float))
    table["CRPS"] = table.CRPS.fillna(table.MAE)
    return deciles, errors, table


LEAD_BUCKETS = [0, 3, 6, 12, 18, 24, 30, 36, 42, 48, 60, 72]


def lead_buckets(forecasts: pd.DataFrame) -> pd.Series:
    labels = [f"{a}–{b}h" for a, b in zip(LEAD_BUCKETS, LEAD_BUCKETS[1:])]
    return pd.cut(forecasts.lead_h, LEAD_BUCKETS, labels=labels).rename("lead")


def updates_to_go(forecasts: pd.DataFrame) -> pd.Series:
    """How many newer NESO vintages were published before each row's hour: 0 for the last one."""
    return forecasts.groupby(level="valid_time").cumcount(ascending=False).rename("updates to go")


def grouped(forecasts: pd.DataFrame, groups: pd.Series, reference: str = "windfor", seed: int = 0) -> pd.DataFrame:
    """Per group and forecast: MAE, and ΔMAE vs `reference` with its 95% day-block bootstrap
    interval, plus CRPS and 10-90% coverage for the probabilistic runs.

    `forecasts` holds one row per (valid_time, issue_time) with `y`, `reference` and packed deciles.
    """
    runs = forecasting.unpack(forecasts)
    errors = pd.DataFrame({reference: forecasts[reference], **{n: d.q50 for n, d in runs.items()}})
    errors = errors.sub(forecasts.y, axis=0)
    days = forecasts.index.get_level_values("valid_time").floor("D")
    rows = {}
    for group in groups.groupby(groups, observed=True).groups:
        part = (groups == group).to_numpy()
        scores = point_scores(errors[part].set_axis(days[part]), reference, seed)
        probabilistic = probabilistic_scores({n: d[part] for n, d in runs.items()}, forecasts.y[part])
        rows[group] = scores[["MAE", "ΔMAE", "lo", "hi"]].join(probabilistic[["CRPS", "cover80"]].astype(float))
    return pd.concat(rows, names=[groups.name, "forecast"])

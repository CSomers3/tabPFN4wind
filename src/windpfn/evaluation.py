"""Scoring, and the committed held-out forecasts it runs on. Everything here is in MW.

The held-out window was scored once, under the protocol frozen at 8e4386d, with ECMWF IFS
weather. TabPFN was then rerun unchanged on the AIFS archive the package reads. Both sets of
forecasts are committed in data/, so the scorecard rebuilds offline without raw pulls or a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from windpfn import forecasting, sources

TEST_START, TEST_END = "2025-01-01", "2026-09-15"

WEATHER = {"ifs": "test_forecasts.parquet", "aifs": "test_forecasts_aifs.parquet"}
ABLATION = "ablation_forecasts.parquet"  # the reference forecasters on the AIFS run's exact inputs
CONTEXT = "context_forecasts.parquet"  # their medians, and TabPFN-3.5's, with the context subsampled
CONTEXT_SIZES = [250, 500, 1000, 2000, 4000]
PROBABILISTIC = [
    "conformal",
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
    references = [pd.read_parquet(sources.DATA / name) for name in ("reference_forecasts.parquet", ABLATION, CONTEXT)]
    return columns.join(forecasting.pack(tabpfn)).join(references)


def scorecard(forecasts: pd.DataFrame, names=PROBABILISTIC):
    """Deciles per model, point errors, and the joint table. A point forecast's CRPS is its MAE."""
    runs = forecasting.unpack(forecasts)
    deciles = {name: runs[name] for name in names}
    medians = {name: forecast.q50 for name, forecast in deciles.items()}
    errors = forecasts.filter(["windfor", "blend"]).join(pd.DataFrame(medians)).sub(forecasts.y, axis=0)
    table = point_scores(errors).join(probabilistic_scores(deciles, forecasts.y).astype(float))
    table["CRPS"] = table.CRPS.fillna(table.MAE)
    return deciles, errors, table


def context_scaling(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE of each forecaster's median by context rows, the full monthly context last."""
    names = ["tabpfn+windfor aifs", *forecasting.REFERENCES]
    full = {name: forecasts[f"{name}|q50"] for name in names}
    rows = {size: {name: forecasts[f"{name} {size}|q50"] for name in names} for size in CONTEXT_SIZES}
    medians = {**rows, "all": full}
    return pd.DataFrame(
        {name: {size: (run[name] - forecasts.y).abs().mean() for size, run in medians.items()} for name in names}
    ).rename_axis("context rows")

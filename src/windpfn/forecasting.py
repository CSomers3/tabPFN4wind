"""Features, TabPFN-3.5, the two reference forecasters, and the monthly backtest driving them.

Every forecaster maps (features, table, train, test) to deciles q10..q90 in MW, where `train`
and `test` are boolean masks over the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from windpfn import dataset

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PERCENTILES = [p / 100 for p in range(1, 100)]

EMBARGO = pd.Timedelta(days=1)
BASELINE_SPLIT = "2024-08-01"

HUB_SHEAR = (100 / 10) ** (1 / 7)  # 10 m wind to 100 m by the one-seventh power law
POWER_CURVE_SPEED = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 32]
POWER_CURVE_OUTPUT = [0, 0, 0.03, 0.08, 0.16, 0.27, 0.41, 0.57, 0.73, 0.86, 0.95, 1, 1, 0]

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 4,
    "verbose": -1,
}


def power_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Capacity factor at each point from its 10 m wind speed, raised to hub height."""
    hub_speed = table[dataset.SPEEDS] * HUB_SHEAR
    return hub_speed.apply(np.interp, xp=POWER_CURVE_SPEED, fp=POWER_CURVE_OUTPUT)


def add_baselines(table: pd.DataFrame) -> pd.DataFrame:
    """Add `powercurve`, a non-negative weighting of the point curves fitted before
    BASELINE_SPLIT, and `blend`, its mean with WINDFOR."""
    curves = power_curve(table).to_numpy()
    fit_rows = table.index < BASELINE_SPLIT
    weights = LinearRegression(positive=True).fit(curves[fit_rows], (table.y / table.cap)[fit_rows])
    powercurve = weights.predict(curves) * table.cap
    return table.assign(powercurve=powercurve, blend=(table.windfor + powercurve) / 2)


def features(table: pd.DataFrame) -> pd.DataFrame:
    """Point power curves, sin/cos of wind direction, capacity, lead and calendar."""
    direction = np.radians(table[dataset.DIRECTIONS])
    return pd.concat(
        [
            power_curve(table).add_prefix("pc_"),
            np.sin(direction).add_prefix("sin_"),
            np.cos(direction).add_prefix("cos_"),
            table[dataset.COVARIATES],
        ],
        axis=1,
    )


def quantile_columns(quantiles) -> list[str]:
    return [f"q{q * 100:.0f}" for q in quantiles]


def _to_mw(capacity_factor: np.ndarray, table: pd.DataFrame, test, quantiles=QUANTILES) -> pd.DataFrame:
    megawatts = capacity_factor * table.cap[test].to_numpy()[:, None]
    return pd.DataFrame(megawatts, table.index[test], quantile_columns(quantiles))


def tabpfn(features, table, train, test) -> pd.DataFrame:
    """TabPFN-3.5 with the `train` rows as its context, predicting capacity factor."""
    from tabpfn_client import TabPFNRegressor  # optional extra; needs an API token

    model = TabPFNRegressor.create_default_for_version("v3.5")
    model.fit(features[train], (table.y / table.cap)[train])
    deciles = model.predict(features[test], output_type="quantiles", quantiles=QUANTILES)
    return _to_mw(np.column_stack(deciles), table, test)


def tabpfn_distribution(features, table, train, test) -> pd.DataFrame:
    """TabPFN-3.5's full predictive distribution, a histogram over capacity factor, read at PERCENTILES."""
    from tabpfn_client import TabPFNRegressor

    model = TabPFNRegressor.create_default_for_version("v3.5")
    model.fit(features[train], (table.y / table.cap)[train])
    full = model.predict(features[test], output_type="full")
    logits = np.asarray(full["logits"], dtype=float)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    cdf = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    cdf = np.concatenate([np.zeros((len(cdf), 1)), cdf], axis=1)
    values = np.array([np.interp(PERCENTILES, row, full["borders"]) for row in cdf])
    return _to_mw(values, table, test, PERCENTILES)


def lightgbm(features, table, train, test, **overrides) -> pd.DataFrame:
    """Quantile LightGBM on capacity factor, one model per decile, sorted so deciles never cross."""
    params = LGBM_PARAMS | overrides
    target = (table.y / table.cap)[train]
    deciles = [
        LGBMRegressor(objective="quantile", alpha=q, **params)
        .fit(features[train], target)
        .predict(features[test])
        for q in QUANTILES
    ]
    return _to_mw(np.sort(np.column_stack(deciles), axis=1), table, test)


def conformal(features, table, train, test, bins: int = 5) -> pd.DataFrame:
    """WINDFOR plus its empirical error deciles within WINDFOR-level bins (Mondrian split-conformal).

    Ignores `features`: its median is WINDFOR recalibrated by level, without weather.
    """
    windfor = table.windfor
    edges = windfor[train].quantile(np.linspace(0, 1, bins + 1)).to_numpy()[1:-1]
    errors = (table.y - windfor)[train]
    offsets = errors.groupby(np.digitize(windfor[train], edges)).quantile(QUANTILES).unstack()
    deciles = windfor[test].to_numpy()[:, None] + offsets.loc[np.digitize(windfor[test], edges)].to_numpy()
    return pd.DataFrame(deciles, table.index[test], quantile_columns(QUANTILES))


CONTEXT_ROWS = 50_000  # one hosted call costs the same up to here


def subsampled(forecaster, rows: int = CONTEXT_ROWS, seed: int = 0):
    """`forecaster` with its context cut to `rows` train rows, sampled uniformly."""

    def forecast(features, table, train, test):
        chosen = np.random.default_rng(seed).choice(np.flatnonzero(train), min(rows, train.sum()), replace=False)
        context = np.zeros(len(table), bool)
        context[chosen] = True
        return forecaster(features, table, context, test)

    return forecast


def backtest(features, table, start, end=None, refit: bool = True, forecaster=tabpfn) -> pd.DataFrame:
    """Forecast [start, end] one calendar month at a time.

    Each month's context is every hour settled EMBARGO before that month's first issue. With
    refit=False the first month's context is kept throughout.
    """
    hours = table.index.to_series()[start:end]
    if refit:
        blocks = [month.index for _, month in hours.groupby(hours.dt.strftime("%Y-%m"))]
    else:
        blocks = [hours.index]
    forecasts = []
    for block in blocks:
        cutoff = table.issue_time.loc[block].min() - EMBARGO
        train = (table.index <= cutoff) & table.y.notna().to_numpy()
        forecasts.append(forecaster(features, table, train, table.index.isin(block)))
    return pd.concat(forecasts)


def pack(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{run name: deciles} -> flat 'run|qNN' columns, for parquet."""
    return pd.concat(
        {f"{name}|{column}": deciles[column] for name, deciles in runs.items() for column in deciles},
        axis=1,
    )


def unpack(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Inverse of pack. Columns without '|' (y, windfor, ...) are ignored."""
    deciles = frame.filter(like="|")
    deciles.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in deciles])
    return {name: deciles[name] for name in deciles.columns.unique(0)}

"""The point-in-time tables, holding only what was public when each forecast was issued.

A forecast is issued PUBLICATION_LAG after a NESO WINDFOR vintage is published, from that vintage,
the newest AIFS run public by then, and the last half-hour of metered wind published by then.
`vintages` holds every vintage; `build` keeps, for each delivery day D, the latest vintage published
by D-1 08:30Z. `check` enforces the timing on every table. Both take their inputs from `pull`,
or from a folder written by `save`: the sample in data/sample/, or the full history in data/history/
that `update` brings up to date for the live forecast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from windpfn import sources, weather

PUBLICATION_LAG = pd.Timedelta("15min")

SPEEDS = [f"ws_{point}" for point in weather.POINTS.index]
DIRECTIONS = [f"wd_{point}" for point in weather.POINTS.index]
COVARIATES = ["cap", "lead_h", "hour", "doy_sin", "doy_cos"]

SAMPLE = sources.DATA / "sample"
SAMPLE_START, SAMPLE_END, SAMPLE_TEST_START = "2026-03-01", "2026-09-15", "2026-08-01"
HISTORY = sources.DATA / "history"
FOLDERS = {"windfor": "bmrs", "metered": "bmrs", "curtailment": "bmrs", "capacity": "bmrs", "aifs": "weather"}
KEYS = {"windfor": ["publishTime", "startTime"], "aifs": ["init_time", "lead_time", "point"]}


def hourly(half_hourly: pd.Series) -> pd.Series:
    """Hour-ending mean: the value at t is the mean of the half-hours starting t-1h and t-30min."""
    first = half_hourly.shift(1, freq="30min")
    second = half_hourly.shift(2, freq="30min")
    return ((first + second) / 2).dropna()


def check(table: pd.DataFrame) -> None:
    """Raise if any input was published after the forecast that used it."""
    leaks = {
        "AIFS run published after issue": ~(table.published <= table.issue_time),
        "WINDFOR vintage not visible at issue": ~(table.windfor_pub + PUBLICATION_LAG <= table.issue_time),
        "metered wind published after issue": ~(table.metered_pub <= table.issue_time),
        "valid_time not after issue": ~(table.index > table.issue_time),
    }
    for message, rows in leaks.items():
        if rows.any():
            raise ValueError(f"{message}: {rows.sum()} rows")


def _window(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    last = pd.Timestamp(end) + pd.Timedelta(days=1) if end else pd.Timestamp.now("UTC").tz_convert(None)
    return pd.Timestamp(start), last


def pull(start=sources.ARCHIVE_START, end=None) -> dict:
    """Every input `vintages(start, end)` reads, from the APIs through the cache in data/raw/."""
    first, last = _window(start, end)
    return {
        "windfor": sources.windfor(first - pd.Timedelta(days=1), last),
        "aifs": weather.pull(first - pd.Timedelta(days=1), last),
        "metered": sources.fuelhh(first - pd.Timedelta(days=2), last),
        "curtailment": sources.curtailment(first - pd.Timedelta(days=2), last, sources.wind_units().index),
        "capacity": sources.capacity(last - sources.B1610_LAG),
    }


def save(inputs: dict, folder: Path = SAMPLE) -> None:
    for name, data in inputs.items():
        path = folder / FOLDERS[name] / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_parquet(path)


def load(folder: Path = SAMPLE) -> dict:
    return {
        name: pd.read_parquet(folder / FOLDERS[name] / f"{name}.parquet").squeeze("columns")
        for name in FOLDERS
    }


def update(inputs: dict) -> dict:
    """`inputs` brought up to now, pulling again from a week before they end, while the APIs may revise."""
    fresh = pull(inputs["metered"].index.max().tz_convert(None).normalize() - sources.REVISABLE)
    merged = {}
    for name, old in inputs.items():
        both = pd.concat([old, fresh[name]])
        if name in KEYS:
            merged[name] = both.drop_duplicates(KEYS[name], keep="last")
        else:
            merged[name] = both[~both.index.duplicated(keep="last")].sort_index()
    return merged


def vintages(start=sources.ARCHIVE_START, end=None, inputs: dict | None = None) -> pd.DataFrame:
    """One row per (WINDFOR vintage published in [start, end], hour it forecasts), indexed by
    valid_time and sorted by it, then by issue_time. MW throughout except the per-point speeds
    (m/s) and directions (deg).

    `y` is available wind, metered output plus curtailment; `windfor` is the vintage's forecast;
    `metered_now` is the last half-hour of metered wind published by the vintage.
    """
    first, last = _window(start, end)
    inputs = inputs or pull(start, end)
    forecasts = inputs["windfor"]
    published = forecasts.publishTime.between(first.tz_localize("UTC"), last.tz_localize("UTC"))
    forecasts = forecasts[published & (forecasts.startTime > forecasts.publishTime + PUBLICATION_LAG)]

    nwp = weather.runs(inputs["aifs"], first - pd.Timedelta(days=1), last)
    public = nwp.groupby(level="run").published.first().reset_index()
    newest = pd.merge_asof(
        forecasts.publishTime.drop_duplicates().sort_values().to_frame(),
        public,
        left_on="publishTime",
        right_on="published",
    )
    forecasts = forecasts.merge(newest, on="publishTime")
    forecasts = forecasts.join(nwp.drop(columns="published"), on=["run", "startTime"], how="inner")

    metered, curtailed = inputs["metered"], inputs["curtailment"]
    target, target_metered = hourly(metered + curtailed), hourly(metered)
    now = pd.DataFrame({"metered_pub": metered.index + pd.Timedelta("30min"), "metered_now": metered.to_numpy()})
    forecasts = pd.merge_asof(
        forecasts.sort_values("publishTime"), now, left_on="publishTime", right_on="metered_pub"
    )

    table = forecasts.assign(
        issue_time=forecasts.publishTime + PUBLICATION_LAG,
        windfor_pub=forecasts.publishTime,
        windfor=forecasts.generation,
        y=forecasts.startTime.map(target),
        y_metered=forecasts.startTime.map(target_metered),
    )
    table = table.rename(columns={"startTime": "valid_time"}).sort_values(["valid_time", "issue_time"])
    table = table.set_index("valid_time")[
        ["issue_time", "run", "published", "windfor_pub", "windfor", "metered_pub", "metered_now",
         "y", "y_metered", *SPEEDS, *DIRECTIONS]
    ]
    valid_time = table.index.to_series()
    day_of_year = 2 * np.pi * table.index.dayofyear / 365.25
    table = table.assign(
        curtailed=table.y - table.y_metered,
        cap=inputs["capacity"].asof(pd.DatetimeIndex(table.issue_time)).to_numpy(),
        lead_h=(valid_time - table.issue_time).dt.total_seconds() / 3600,
        hour=table.index.hour,
        doy_sin=np.sin(day_of_year),
        doy_cos=np.cos(day_of_year),
    )
    check(table)
    return table


def build(start=sources.ARCHIVE_START, end=None, inputs: dict | None = None) -> pd.DataFrame:
    """The day-ahead table: the hours of each delivery day D in [start, end], from the latest
    vintage published by D-1 08:30Z, one row per hour. Columns as in `vintages`."""
    first = pd.Timestamp(start) - pd.Timedelta(days=1)
    inputs = inputs or pull(first, end)
    table = vintages(first, end, inputs)
    issued = table.index.floor("D").map(sources.issue_times(inputs["windfor"], start, end))
    return table[table.windfor_pub.values == issued.values]

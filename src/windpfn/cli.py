"""Command-line entry points.

    windpfn-fetch       pull and cache every raw input into data/raw/ (hours the first time)
    windpfn-backtest    regenerate held-out forecasts into data/results/
    windpfn-score       print the README tables from the committed forecasts, offline
    windpfn-page        export the data behind notebooks/explorer.py into notebooks/public/
    windpfn-live        issue the full distribution after NESO's newest update, into notebooks/public/live.csv,
                        from data/history/ and what has been published since
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from windpfn import dataset, evaluation, forecasting, sources, weather

RESULTS = sources.DATA / "results"
PUBLIC = sources.ROOT / "notebooks/public"
LIVE = PUBLIC / "live.csv"
HELD_OUT = {
    "windfor": "NESO",
    "conformal": "NESO recalibrated + conformal interval",
    "tabpfn+windfor ifs": "TabPFN-3.5 · IFS",
    "tabpfn+windfor aifs": "TabPFN-3.5 · AIFS",
}
ABLATION = {
    "tabpfn+windfor aifs": "TabPFN-3.5",
    **{name: label for name, (label, _) in forecasting.REFERENCES.items()},
}


def _parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)


def fetch() -> None:
    """Pull and cache every raw input for delivery days [start, end] into data/raw/.

    With --out, also write what the pipeline reads to that folder, as data/sample/ was made:

        windpfn-fetch --start 2026-02-28 --end 2026-09-15 --out data/sample
    """
    parser = _parser(fetch.__doc__)
    parser.add_argument("--start", default=sources.ARCHIVE_START, help="first delivery day")
    parser.add_argument("--end", help="last delivery day (default: today)")
    parser.add_argument("--out", type=Path, help="folder to write the pulled inputs to")
    args = parser.parse_args()

    inputs = dataset.pull(args.start, args.end)
    if args.out:
        dataset.save(inputs, args.out)


def _run_tabpfn(table, start, end):
    features = forecasting.features(table)
    runs = {}
    for name, inputs in {"tabpfn": features, "tabpfn+windfor": features.join(table.windfor)}.items():
        runs[name] = forecasting.backtest(inputs, table, start, end)
        runs[f"{name} frozen"] = forecasting.backtest(inputs, table, start, end, refit=False)
    columns = ["y", "windfor", "blend", "lead_h", "cap"]
    return table.loc[start:end, columns].join(forecasting.pack(runs))


def _run_references(table, start, end):
    features = forecasting.features(table)
    with_windfor = features.join(table.windfor)
    runs = {
        "lgbm": forecasting.backtest(features, table, start, end, forecaster=forecasting.lightgbm),
        "lgbm+windfor": forecasting.backtest(with_windfor, table, start, end, forecaster=forecasting.lightgbm),
        "conformal": forecasting.backtest(features, table, start, end, forecaster=forecasting.conformal),
    }
    return forecasting.pack(runs)


def _run_ablation(table, start, end):
    """The reference forecasters on TabPFN's exact inputs: weather + WINDFOR."""
    inputs = forecasting.features(table).join(table.windfor)
    runs = {
        name: forecasting.backtest(inputs, table, start, end, forecaster=forecaster)
        for name, (_, forecaster) in forecasting.REFERENCES.items()
    }
    return forecasting.pack(runs)


def _run_context(table, start, end):
    """TabPFN-3.5 and the reference forecasters with each month's context cut to CONTEXT_SIZES rows."""
    inputs = forecasting.features(table).join(table.windfor)
    forecasters = {"tabpfn+windfor aifs": forecasting.tabpfn} | {
        name: forecaster for name, (_, forecaster) in forecasting.REFERENCES.items()
    }
    runs = {
        f"{name} {rows}": forecasting.backtest(
            inputs, table, start, end, forecaster=forecasting.subsampled(forecaster, rows)
        )
        for rows in evaluation.CONTEXT_SIZES
        for name, forecaster in forecasters.items()
    }
    return forecasting.pack(runs)


SUBJECTS = {"tabpfn": _run_tabpfn, "references": _run_references, "ablation": _run_ablation, "context": _run_context}


def backtest() -> None:
    """Regenerate held-out forecasts into data/results/, never over the committed ones in data/.

    Reads the committed inputs in data/history/. `tabpfn` runs weather-only and weather + WINDFOR
    on the day-ahead table, each refit monthly and frozen, on the hosted API (needs a tabpfn-client
    token). `references` runs LightGBM and conformal, `ablation` LightGBM and a quantile regression
    forest on TabPFN's inputs, both locally. `context` reruns TabPFN-3.5 (hosted) and those two with
    the context subsampled.
    """
    parser = _parser(backtest.__doc__)
    parser.add_argument("subject", choices=list(SUBJECTS))
    parser.add_argument("--start", default=evaluation.TEST_START)
    parser.add_argument("--end", default=evaluation.TEST_END)
    args = parser.parse_args()

    inputs = dataset.load(dataset.HISTORY)
    table = forecasting.add_baselines(dataset.build(sources.ARCHIVE_START, args.end, inputs))
    forecasts = SUBJECTS[args.subject](table, args.start, args.end)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.subject}_{args.start}_{args.end}.parquet"
    forecasts.to_parquet(out)
    print(f"{out}: {len(forecasts)} hours")


def score() -> None:
    """Print the README's held-out tables from the committed forecasts. Offline: no raw data, no token."""
    _parser(score.__doc__).parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # "ΔMAE" on a Windows console
    forecasts = evaluation.load_forecasts()
    _, errors, table = evaluation.scorecard(forecasts, [*list(HELD_OUT)[1:], *list(ABLATION)[1:]])
    days = forecasts.index.floor("D").nunique()
    print(f"{forecasts.index[0]:%Y-%m-%d} to {forecasts.index[-1]:%Y-%m-%d}: {days} days, {len(forecasts)} hours, MW")
    columns = ["MAE", "ΔMAE", "lo", "hi", "CRPS", "cover80", "winkler80"]
    print("\nResults\n" + table.loc[list(HELD_OUT), columns].rename(index=HELD_OUT).round(3).to_string())
    versus = evaluation.point_scores(errors[list(ABLATION)], reference="tabpfn+windfor aifs")[["ΔMAE", "lo", "hi"]]
    ablation = table.loc[list(ABLATION), ["MAE", "CRPS", "cover80", "winkler80"]].join(versus.add_suffix(" vs TabPFN"))
    print("\nAblation, identical AIFS inputs\n" + ablation.rename(index=ABLATION).round(3).to_string())
    scaling = evaluation.context_scaling(forecasts).rename(columns=ABLATION)
    print("\nMAE by context rows\n" + scaling.round(0).to_string())


def page() -> None:
    """Export the data behind notebooks/explorer.py, which runs under Pyodide without windpfn.

    Writes notebooks/public/meta.json: the wind units whose curtailment the page adds to live outturn,
    and the map of wind farms and weather points.
    `windpfn-live` writes live.csv beside it.
    """
    _parser(page.__doc__).parse_args()
    countries = sources.get_json("naturalearth", weather.NATURAL_EARTH)["features"]
    outline = [
        [[round(lon, 2), round(lat, 2)] for lon, lat in polygon[0]]
        for country in countries
        if country["properties"]["ADM0_A3"] in ("GBR", "IRL")
        for polygon in country["geometry"]["coordinates"]
    ]
    farms = weather.farms()
    lon, lat = weather.to_lonlat(farms.x.values, farms.y.values)
    points = weather.POINTS

    meta = {
        "units": sorted(sources.wind_units().index),
        "outline": outline,
        "farms": [[round(x, 3), round(y, 3), round(mw)] for x, y, mw in zip(lon, lat, farms.mw)],
        "points": [[p.lon, p.lat, p.mw, name.startswith("off")] for name, p in points.iterrows()],
    }
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(PUBLIC / "meta.json")


LIVE_DAYS = 14


def live() -> None:
    """Issue TabPFN-3.5's full distribution, as p1–p99, for every hour of NESO's newest WINDFOR
    vintage, into notebooks/public/live.csv.

    Run after each of NESO's eight daily vintages; a vintage already issued is skipped. Hours not
    yet started take the new forecast, and started hours keep the last one issued before them, so
    each row is the most recent forecast for its hour. The context is data/history/ plus everything
    published since, pulled into data/raw/. Needs a tabpfn-client token.
    """
    _parser(live.__doc__).parse_args()
    record = pd.read_csv(LIVE, index_col="hour", parse_dates=["hour", "issue_time"]) if LIVE.exists() else None
    newest = sources.newest_windfor()
    if record is not None and record.issue_time.max() >= (newest + dataset.PUBLICATION_LAG).tz_convert(None):
        print(f"NESO's {newest:%Y-%m-%d %H:%M} UTC update already issued")
        return

    table = dataset.vintages(sources.ARCHIVE_START, inputs=dataset.update(dataset.load(dataset.HISTORY)))
    issue = table.issue_time.max()

    inputs = forecasting.features(table).assign(
        windfor=table.windfor.to_numpy(), metered_now=table.metered_now.to_numpy()
    )
    train = (table.index <= issue - forecasting.EMBARGO) & table.y.notna().to_numpy()
    test = (table.issue_time == issue).to_numpy()
    percentiles = forecasting.subsampled(forecasting.tabpfn_distribution)(inputs, table, train, test)
    issued = table.loc[test, ["issue_time", "windfor"]].join(percentiles.round())
    issued.index = issued.index.tz_convert(None)
    issued["issue_time"] = issued.issue_time.dt.tz_convert(None)

    if record is not None:
        first = issued.index[0]
        issued = pd.concat([record[first - pd.Timedelta(days=LIVE_DAYS) : first - pd.Timedelta("1h")], issued])
    columns = ["windfor", *forecasting.quantile_columns(forecasting.PERCENTILES)]
    issued = issued.astype({column: "int64" for column in columns}).rename_axis("hour")
    issued.to_csv(LIVE, date_format="%Y-%m-%d %H:%M")
    print(f"Forecast from NESO's {issue - dataset.PUBLICATION_LAG:%Y-%m-%d %H:%M} UTC update")

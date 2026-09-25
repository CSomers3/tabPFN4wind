"""Elexon BMRS and NESO pulls, cached under data/raw/ with a manifest of every response's sha256."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
DATASETS = f"{ELEXON}/datasets"

ARCHIVE_START = "2024-04-02"  # first day forecast from an AIFS run on dynamical.org
ISSUE_CUTOFF = pd.Timedelta(hours=8, minutes=30)
B1610_LAG = pd.Timedelta(days=7)
REVISABLE = pd.Timedelta(days=7)

_APPEND_LOCK = threading.Lock()


def request(url: str, method: str = "GET", tries: int = 6, **kwargs) -> requests.Response:
    """requests.request with exponential backoff on connection errors, 429 and 5xx."""
    for attempt in range(tries):
        if attempt:
            time.sleep(2**attempt)
        try:
            response = requests.request(method, url, timeout=120, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == tries - 1:
                raise
            continue
        if response.status_code != 429 and response.status_code < 500:
            break
    response.raise_for_status()
    return response


def write_atomic(path: Path, body: bytes) -> None:
    """Write through a temporary file, so an interrupted pull never leaves a truncated cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    partial.write_bytes(body)
    partial.replace(path)


def append_jsonl(path: Path, records: list[dict]) -> None:
    with _APPEND_LOCK, path.open("a") as log:
        log.writelines(json.dumps(record) + "\n" for record in records)


def get_json(source: str, url: str, params: dict | None = None, refresh: bool = False):
    """A cached GET. Pass `refresh` for windows the API may still revise or extend."""
    full_url = url + ("?" + urlencode(sorted(params.items())) if params else "")
    key = hashlib.sha256(full_url.encode()).hexdigest()[:24]
    path = RAW / source / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_bytes())

    body = request(url, params=params).content
    write_atomic(path, body)
    record = {
        "source": source,
        "key": key,
        "url": full_url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }
    append_jsonl(RAW / "manifest.jsonl", [record])
    return json.loads(body)


def _utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def _today() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _recent(end) -> bool:
    end = pd.Timestamp(end)
    return (end.tz_convert(None) if end.tzinfo else end) >= _today() - REVISABLE


def _months(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).to_period("M").start_time, end, freq="MS")


def windfor(start="2016-01-01", end=None) -> pd.DataFrame:
    """Every WINDFOR vintage published in [start, end]: one row per (publishTime, startTime)."""
    end = pd.Timestamp(end) if end else _today() + pd.Timedelta(days=1)
    pulls = []
    for month in _months(start, end):
        next_month = month + pd.offsets.MonthBegin()
        params = {
            "publishDateTimeFrom": f"{month:%Y-%m-%dT%H:%MZ}",
            "publishDateTimeTo": f"{next_month:%Y-%m-%dT%H:%MZ}",
        }
        url = f"{DATASETS}/WINDFOR/stream"
        pulls.append(pd.DataFrame(get_json("windfor", url, params, refresh=_recent(next_month))))
    vintages = pd.concat(pulls).drop_duplicates()
    return vintages.assign(publishTime=_utc(vintages.publishTime), startTime=_utc(vintages.startTime))


def _fuelhh(start, end, **params) -> pd.DataFrame:
    """FUELHH rows, keeping the latest publication of each (startTime, fuelType).

    Keyed on UTC startTime: the API's (settlementDate, period) labels are wrong on clock-change
    days until 2022-07.
    """
    end = pd.Timestamp(end) if end else _today()
    pulls = []
    for month in _months(start, end):
        last = min(month + pd.offsets.MonthEnd(0), end)
        window = {"settlementDateFrom": f"{month:%Y-%m-%d}", "settlementDateTo": f"{last:%Y-%m-%d}"}
        url = f"{DATASETS}/FUELHH/stream"
        pulls.append(pd.DataFrame(get_json("fuelhh", url, window | params, refresh=_recent(last))))
    rows = pd.concat(pulls)
    rows = rows.assign(startTime=_utc(rows.startTime), publishTime=_utc(rows.publishTime))
    return rows.sort_values("publishTime").drop_duplicates(["startTime", "fuelType"], keep="last")


def fuelhh(start="2016-01-01", end=None) -> pd.Series:
    """Metered transmission wind (MW) per half-hour startTime."""
    return _fuelhh(start, end, fuelType="WIND").set_index("startTime").generation.sort_index()


def generation_mix(start, end) -> pd.DataFrame:
    """Metered transmission generation (MW) per half-hour startTime, one column per fuel type."""
    mix = _fuelhh(start, end)
    return mix.pivot(index="startTime", columns="fuelType", values="generation").sort_index()


def curtailment(start, end, units) -> pd.Series:
    """MW removed from `units` by accepted balancing-mechanism bids net of offers (BOAV).

    Zero where nothing was accepted. Self-curtailment at negative prices is not a BM action and
    is missing.
    """

    def one_day(day):
        volumes = pd.concat(
            pd.DataFrame(
                get_json(
                    "boav",
                    f"{ELEXON}/balancing/settlement/acceptance/volumes/all/{side}/{day:%Y-%m-%d}",
                    refresh=_recent(day),
                )["data"]
            )
            for side in ("bid", "offer")
        )
        return volumes[volumes.bmUnit.isin(units)] if len(volumes) else None

    with ThreadPoolExecutor(8) as pool:
        volumes = pd.concat(pool.map(one_day, pd.date_range(start, end or _today())))
    removed = -2 * volumes.groupby(_utc(volumes.startTime)).totalVolumeAccepted.sum()
    half_hours = pd.date_range(_utc(start), removed.index.max(), freq="30min")
    return removed.reindex(half_hours, fill_value=0.0).rename("curtailment")


def prices(start, end=None) -> pd.Series:
    """Day-ahead market index price (APX, £/MWh) per half-hour startTime."""
    pulls = []
    for week in pd.date_range(start, end or _today(), freq="7D"):
        week_end = week + pd.Timedelta(days=7)
        params = {"from": f"{week:%Y-%m-%d}", "to": f"{week_end:%Y-%m-%d}", "dataProviders": "APXMIDP"}
        url = f"{DATASETS}/MID/stream"
        pulls.append(pd.DataFrame(get_json("mid", url, params, refresh=_recent(week_end))))
    price = pd.concat(pulls)
    price = price.assign(startTime=_utc(price.startTime)).drop_duplicates("startTime")
    return price.set_index("startTime").price.sort_index()


def system_prices(start, end=None) -> pd.Series:
    """Imbalance price (£/MWh) per half-hour startTime. GB has had a single price since 2015."""

    def one_day(day):
        url = f"{ELEXON}/balancing/settlement/system-prices/{day:%Y-%m-%d}"
        rows = get_json("disebsp", url, refresh=_recent(day))["data"]
        return pd.DataFrame(rows) if rows else None

    with ThreadPoolExecutor(8) as pool:
        price = pd.concat(pool.map(one_day, pd.date_range(start, end or _today())))
    price = price.assign(startTime=_utc(price.startTime)).drop_duplicates("startTime")
    return price.set_index("startTime").systemSellPrice.sort_index()


NESO_BALANCING_COSTS = (
    "527a5f40-942b-416b-99df-81a51c30d041",  # FY 2024/25
    "46183ba7-48df-4318-9b4b-06828348d46e",  # FY 2025/26
)


def balancing_costs() -> pd.DataFrame:
    """NESO's daily balancing costs (£) by category."""
    categories = ["Energy Imbalance", "Frequency Control", "Positive Reserve", "Negative Reserve",
                  "Constraints", "Other"]
    sums = ", ".join(f'SUM("{name}") AS "{name}"' for name in categories)
    pulls = []
    for resource in NESO_BALANCING_COSTS:
        sql = f'SELECT "SETT_DATE", {sums} FROM "{resource}" GROUP BY "SETT_DATE"'
        rows = get_json("neso_costs", "https://api.neso.energy/api/3/action/datastore_search_sql",
                        {"sql": sql})
        pulls.append(pd.DataFrame(rows["result"]["records"]))
    costs = pd.concat(pulls).set_index("SETT_DATE")
    costs.index = pd.to_datetime(costs.index)
    return costs.astype(float).sort_index()


def issue_times(vintages: pd.DataFrame, start, end=None, cutoff=ISSUE_CUTOFF) -> pd.Series:
    """Delivery day D (UTC) -> publishTime of the latest WINDFOR vintage at or before D-1 cutoff."""
    published = pd.Series(vintages.publishTime.unique()).sort_values()
    days = pd.date_range(_utc(start), published.max().floor("D") + pd.Timedelta(days=1), freq="D")
    cutoffs = pd.DataFrame({"t": days - pd.Timedelta(days=1) + cutoff, "day": days})
    matched = pd.merge_asof(cutoffs, pd.DataFrame({"t": published, "issue": published}), on="t")
    issues = matched.set_index("day").issue.dropna()
    return issues[: _utc(end)] if end else issues


def day_ahead_vintage(vintages: pd.DataFrame, cutoff=ISSUE_CUTOFF) -> pd.DataFrame:
    """For each hour, the latest vintage published by D-1 cutoff, keeping future hours only."""
    ahead = vintages[vintages.startTime > vintages.publishTime]
    day = ahead.startTime.dt.floor("D")
    ahead = ahead[ahead.publishTime <= day - pd.Timedelta(days=1) + cutoff]
    latest = ahead.sort_values("publishTime").drop_duplicates("startTime", keep="last")
    return latest.set_index("startTime")


def wind_units() -> pd.Series:
    """Nameplate MW of transmission wind BM units (T_, E_). A current snapshot, undated."""
    units = pd.DataFrame(get_json("bmunits", f"{ELEXON}/reference/bmunits/all"))
    units = units[(units.fuelType == "WIND") & units.elexonBmUnit.str[:2].isin(["T_", "E_"])]
    units = units.drop_duplicates("elexonBmUnit").set_index("elexonBmUnit")
    return units.generationCapacity.astype(float)


def first_output(start="2023-06-07", end=None) -> pd.Series:
    """BM unit -> first weekly B1610 sample (Wednesday 11:00Z) with positive metered output."""
    nameplate, first_seen = wind_units(), {}
    for week in pd.date_range(start, end or _today(), freq="7D"):
        params = {"from": f"{week:%Y-%m-%d}T11:00Z", "to": f"{week:%Y-%m-%d}T11:05Z"}
        sample = pd.DataFrame(
            get_json("b1610", f"{DATASETS}/B1610/stream", params, refresh=_recent(week))
        )
        if sample.empty:
            continue
        running = sample[sample.bmUnit.isin(nameplate.index) & (sample.quantity > 0)].bmUnit
        for unit in running:
            first_seen.setdefault(unit, pd.Timestamp(week, tz="UTC"))
    return pd.Series(first_seen, name="first_output")


def capacity(end=None) -> pd.Series:
    """MW of wind units whose first metered output had been published, stepping up at each publication."""
    known_from = (first_output(end=end) + B1610_LAG).sort_values()
    steps = pd.Series(wind_units()[known_from.index].cumsum().to_numpy(), pd.DatetimeIndex(known_from))
    return steps.groupby(level=0).last().rename("capacity")


def poll_windfor_latency(hours: float = 3.0, out: Path = DATA / "audit/windfor_first_seen.jsonl"):
    """Log when each WINDFOR publishTime first appears on the API.

    One try per poll, so a slow response is never stamped late.
    """
    stop, seen = pd.Timestamp.now("UTC") + pd.Timedelta(hours=hours), set()
    while (now := pd.Timestamp.now("UTC")) < stop:
        params = {
            "publishDateTimeFrom": f"{now - pd.Timedelta(hours=12):%Y-%m-%dT%H:%MZ}",
            "publishDateTimeTo": f"{now + pd.Timedelta(hours=2):%Y-%m-%dT%H:%MZ}",
        }
        try:
            response = request(f"{DATASETS}/WINDFOR/stream", tries=1, params=params)
            published = {row["publishTime"] for row in response.json()}
        except requests.RequestException as error:
            print("error", error, flush=True)
            published = set()
        records = [
            {"publishTime": stamp, "first_seen_utc": now.isoformat(timespec="seconds"), "initial": not seen}
            for stamp in sorted(published - seen)
        ]
        append_jsonl(out, records)
        seen |= published
        time.sleep(120)

"""ECMWF IFS 0.25° open data (CC-BY-4.0) at the NWP points, timed by actual publication."""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import eccodes
import numpy as np
import pandas as pd
import requests

from .cache import DATA

BASES = ["https://storage.googleapis.com/ecmwf-open-data",  # preferred for data; AWS throttles bulk GETs
         "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]
CACHE = DATA / "raw/ecmwf_open"
POINTS = pd.read_csv(DATA / "points.csv").set_index("point")[["lat", "lon"]]
_LOCK = threading.Lock()  # eccodes is not thread-safe


def _url(run, step, base, stream):
    return f"{base}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/{run:%Y%m%d%H}0000-{step}h-{stream}-fc"


def _get(url, method="GET", tries=8, **kw):
    for k in range(tries):
        try:
            r = requests.request(method, url, timeout=60, **kw)
            if r.status_code not in (429, 500, 503):
                break
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
        time.sleep(2 ** k)
    r.raise_for_status()
    return r


def locate(run, step) -> list[tuple[str, str, pd.Timestamp]]:
    """Existing copies (base, stream, upload time) across mirrors and stream names (06/18Z: scda, later oper)."""
    out = []
    for base in BASES:
        for st in ("oper",) if run.hour in (0, 12) else ("scda", "oper"):
            u = _url(run, step, base, st)
            try:
                h = _get(u + ".grib2", "HEAD", tries=8 if base == BASES[0] else 3).headers
                out.append((base, st, pd.Timestamp(parsedate_to_datetime(h["Last-Modified"]))))
            except requests.HTTPError:
                continue
    return out


def steps_for(run, D):
    first = int((D - run) / pd.Timedelta("1h"))
    return list(range(first, first + 25, 3))


def runs_for(issue: pd.Timestamp, D: pd.Timestamp):
    """Runs public (earliest mirror upload) at or before `issue` and on GCS, newest first."""
    for k in range(8):
        run = issue.floor("6h") - k * pd.Timedelta("6h")
        loc = locate(run, steps_for(run, D)[-1])
        gcs = [st for base, st, _ in loc if base == BASES[0]]
        if gcs and min(t for *_, t in loc) <= issue:
            yield run, min(t for *_, t in loc), gcs[0]


def run_for(issue, D):
    return next(runs_for(issue, D))


def _fields(url, params=("100u", "100v")):
    idx = {m["param"]: m for m in map(json.loads, _get(url + ".index").text.splitlines())}
    return [_field(url, idx[p]) for p in params]


def _field(url, m):
    rng = f"bytes={m['_offset']}-{m['_offset'] + m['_length'] - 1}"
    b = _get(url + ".grib2", headers={"Range": rng}).content
    with _LOCK:
        h = eccodes.codes_new_from_message(b)
        g = {k: eccodes.codes_get(h, k) for k in ["Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                                                 "longitudeOfFirstGridPointInDegrees", "iDirectionIncrementInDegrees"]}
        v = eccodes.codes_get_values(h).reshape(g["Nj"], g["Ni"])
        eccodes.codes_release(h)
    d = g["iDirectionIncrementInDegrees"]
    i = np.rint((g["latitudeOfFirstGridPointInDegrees"] - POINTS.lat) / d).astype(int)
    j = np.rint((POINTS.lon - g["longitudeOfFirstGridPointInDegrees"]) / d).astype(int) % g["Ni"]
    return v[i, j], {"url": url + ".grib2", "range": rng, "sha256": hashlib.sha256(b).hexdigest()}


def fetch_for(issue: pd.Timestamp, D: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hourly 100 m speed/direction at POINTS for delivery day D from the run public at `issue`."""
    D = D if D is not None else issue.floor("D") + pd.Timedelta(days=1)
    path = CACHE / f"{D:%Y%m%d}_{issue:%H%M}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    for run, pub, stream in runs_for(issue, D):  # older run if a step file is missing on GCS
        try:
            rows, src = [], []
            for s in steps_for(run, D):
                (u, a), (v, b) = _fields(_url(run, s, BASES[0], stream))
                rows.append(pd.DataFrame({"point": POINTS.index, "valid_time": run + pd.Timedelta(hours=s), "u": u, "v": v}))
                src += [a, b]
            break
        except requests.HTTPError as e:
            if e.response.status_code != 404:
                raise
    else:
        raise LookupError(f"no complete run for {D:%Y-%m-%d}")
    uv = pd.concat(rows).pivot(index="valid_time", columns="point")
    uv = uv.resample("1h").interpolate()[D:D + pd.Timedelta("23h")]
    ws, wd = np.hypot(uv.u, uv.v), (270 - np.degrees(np.arctan2(uv.v, uv.u))) % 360
    out = pd.concat({"wind_speed_100m": ws.stack(), "wind_direction_100m": wd.stack()}, axis=1).reset_index()
    out = out.assign(run=run, published=pub)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    with (CACHE / "manifest.jsonl").open("a") as f:
        f.writelines(json.dumps(x) + "\n" for x in src)
    return out


def backfill(issues: pd.Series, workers: int = 16):
    def one(a):
        try:
            fetch_for(*a)
        except Exception as e:
            print(a[1], e, flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, [(i, D) for D, i in issues.items()]))

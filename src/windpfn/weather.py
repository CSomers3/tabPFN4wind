"""ECMWF AIFS Single forecasts from dynamical.org, at the 20 wind-capacity points in data/points.csv.

dynamical.org keeps no publication time per run. Its update history shows each run landing 6h14m
after initialisation, so a run is taken as public PUBLISHED_AFTER its init time. The wind is at 10 m:
dynamical.org's AIFS archive carries 100 m only from 2025-02-24.
The points are capacity-weighted k-means clusters of REPD wind farms (`clusters`).
"""

from __future__ import annotations

import dynamical_catalog
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from sklearn.cluster import KMeans

from windpfn import sources

AIFS = "ecmwf-aifs-single-forecast"
FIRST_RUN = pd.Timestamp("2024-04-01")
PUBLISHED_AFTER = pd.Timedelta("6h30min")
MAX_LEAD = pd.Timedelta(hours=90)
CACHE = sources.RAW / "aifs"
POINTS = pd.read_csv(sources.DATA / "points.csv").set_index("point")

REPD = sources.RAW / "repd/REPD_Q2_2026.csv"
REPD_URL = (
    "https://assets.publishing.service.gov.uk/media/6a6cbdc00c36759b5ccaa305/"
    "REPD_Publication_Q2_2026.csv"
)

# REPD lists these as under construction, but they already meter in B1610.
GENERATING_EARLY = {"Dogger Bank A & B": "2023-11-01", "Sofia": "2026-04-22"}
MIN_FARM_MW = 10
CLUSTER_PREFIXES = {"Wind Offshore": "off", "Wind Onshore": "on"}
CLUSTERS_PER_TECH = 10

to_lonlat = Transformer.from_crs(27700, 4326, always_xy=True).transform


def farms() -> pd.DataFrame:
    """Operational GB wind farms of at least MIN_FARM_MW, British National Grid coordinates."""
    if not REPD.exists():
        sources.write_atomic(REPD, sources.request(REPD_URL).content)
    repd = pd.read_csv(REPD, encoding="latin-1")
    status, tech = repd["Development Status (short)"], repd["Technology Type"]
    early = repd["Site Name"].str.startswith(tuple(GENERATING_EARLY), na=False)
    operational_wind = (status == "Operational") & tech.str.startswith("Wind", na=False)
    repd = repd[(operational_wind | early) & (repd.Country != "Northern Ireland")]
    repd = repd.rename(
        columns={
            "Site Name": "name",
            "Installed Capacity (MWelec)": "mw",
            "X-coordinate": "x",
            "Y-coordinate": "y",
            "Operational": "operational",
            "Technology Type": "tech",
        }
    )
    sites = repd[["name", "tech", "mw", "x", "y", "operational"]].copy()
    sites["name"] = sites.name.str.strip(" \xa0")
    for column in ("mw", "x", "y"):
        sites[column] = pd.to_numeric(sites[column], errors="coerce")
    sites["operational"] = pd.to_datetime(
        sites.operational, dayfirst=True, errors="coerce"
    ).dt.tz_localize("UTC")
    for name, date in GENERATING_EARLY.items():
        sites.loc[sites.name.str.startswith(name), "operational"] = pd.Timestamp(date, tz="UTC")
    sites = sites.dropna(subset=["mw", "x", "y"])
    return sites[sites.mw >= MIN_FARM_MW].reset_index(drop=True)


def clusters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Farms labelled with a `point`, and the capacity-weighted centroid of each point."""
    sites = farms()
    for j, (tech, prefix) in enumerate(CLUSTER_PREFIXES.items()):
        mask = sites.tech == tech
        kmeans = KMeans(CLUSTERS_PER_TECH, n_init=20, random_state=0)
        labels = kmeans.fit(sites[mask][["x", "y"]], sample_weight=sites.mw[mask]).labels_
        sites.loc[mask, "point"] = [f"{prefix}_{CLUSTERS_PER_TECH * j + i:02d}" for i in labels]

    def centroid(group):
        return pd.Series(
            {
                "x": (group.x * group.mw).sum() / group.mw.sum(),
                "y": (group.y * group.mw).sum() / group.mw.sum(),
                "mw": group.mw.sum(),
                "n": len(group),
                "top": group.nlargest(1, "mw").name.iloc[0],
            }
        )

    points = sites.groupby("point").apply(centroid, include_groups=False)
    points["lon"], points["lat"] = to_lonlat(points.x.values, points.y.values)
    return sites, points.round({"lat": 3, "lon": 3, "mw": 0})


def _month(month: pd.Timestamp) -> pd.DataFrame:
    """10 m u and v (m/s) at POINTS from the AIFS runs initialised in `month`, as dynamical.org
    serves them: one row per (init_time, lead_time, point), 6-hourly out to MAX_LEAD. Cached once
    the month's last run is public."""
    path = CACHE / f"{month:%Y-%m}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    next_month = month + pd.offsets.MonthBegin()
    wind = dynamical_catalog.open(AIFS, chunks=None)[["wind_u_10m", "wind_v_10m"]]
    wind = wind.sel(init_time=slice(month, next_month - pd.Timedelta("1s")), lead_time=slice(None, MAX_LEAD))
    at_points = wind.sel(
        latitude=xr.DataArray(POINTS.lat, dims="point"),
        longitude=xr.DataArray(POINTS.lon, dims="point"),
        method="nearest",
    )
    uv = at_points.load().to_dataframe()[["wind_u_10m", "wind_v_10m"]].reset_index()
    if pd.Timestamp.now("UTC").tz_convert(None) > next_month + PUBLISHED_AFTER:
        sources.write_atomic(path, uv.to_parquet())
    return uv


def pull(start, end) -> pd.DataFrame:
    """10 m u and v at POINTS from the AIFS runs initialised in every month spanning [start, end]."""
    start = max(pd.Timestamp(start), FIRST_RUN)
    return pd.concat(map(_month, pd.date_range(start.to_period("M").start_time, end, freq="MS")))


def runs(raw: pd.DataFrame, start, end) -> pd.DataFrame:
    """Hourly 10 m wind speed `ws_<point>` (m/s) and direction `wd_<point>` (deg) from every run in
    `raw` initialised in [start, end], out to MAX_LEAD, indexed by (run, valid_time), with the time
    each run was `published`. Runs not yet complete to MAX_LEAD are left out."""
    start = max(pd.Timestamp(start), FIRST_RUN)
    uv = raw.pivot(index=["init_time", "lead_time"], columns="point").loc[start:end]
    uv = uv[uv.notna().all(axis=1).groupby(level="init_time").transform("all")]
    uv = uv.groupby(level="init_time").apply(lambda run: run.droplevel("init_time").resample("1h").interpolate())
    u, v = uv["wind_u_10m"], uv["wind_v_10m"]

    wind = pd.concat(
        [np.hypot(u, v).add_prefix("ws_"), ((270 - np.degrees(np.arctan2(v, u))) % 360).add_prefix("wd_")],
        axis=1,
    ).rename_axis(columns=None)
    run = uv.index.get_level_values(0).tz_localize("UTC")
    wind.index = pd.MultiIndex.from_arrays([run, run + uv.index.get_level_values(1)], names=["run", "valid_time"])
    return wind.assign(published=run + PUBLISHED_AFTER)

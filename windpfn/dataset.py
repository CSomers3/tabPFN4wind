"""One row per delivery hour: features known at issue, the WINDFOR benchmark, and the target."""
import numpy as np
import pandas as pd
from . import capacity, data, nwp

PUB_LAG = pd.Timedelta("15min")  # WINDFOR vintage visible ~2 min after publishTime; issue after that
WS = [f"ws_{p}" for p in nwp.POINTS.index]
WD = [f"wd_{p}" for p in nwp.POINTS.index]
FEATURES = WS + ["cap", "lead_h", "hour", "doy_sin", "doy_cos"]  # claim (b) adds "windfor"


def hourly(o: pd.Series) -> pd.Series:
    """Hour-ending mean: value at t = mean of half-hours starting t-1h, t-30m."""
    return ((o.shift(1, freq="30min") + o.shift(2, freq="30min")) / 2).dropna()


def check(df: pd.DataFrame):
    assert (df.published <= df.issue_time).all(), "NWP run published after issue"
    assert (df.windfor_pub + PUB_LAG <= df.issue_time).all(), "WINDFOR vintage not yet visible at issue"
    assert (df.valid_time > df.issue_time).all(), "valid_time not in the future"


def build(start="2024-03-16", end=None) -> pd.DataFrame:
    iss = data.issues(start, end)
    pad = lambda days: end and pd.Timestamp(end) + pd.Timedelta(days=days)
    o = data.fuelhh(pd.Timestamp(start) - pd.Timedelta(days=1), pad(1))
    c = data.curtailment(pd.Timestamp(start) - pd.Timedelta(days=1), pad(1), capacity.units().index)
    w = data.windfor(pd.Timestamp(start) - pd.Timedelta(days=2), pad(1))
    y, y_met = hourly(o + c), hourly(o)
    rows = []
    for D, tau in iss.items():  # tau: WINDFOR vintage; NWP must be public by tau
        f = nwp.fetch_for(tau, D)
        X = pd.concat([f.pivot(index="valid_time", columns="point", values=f"wind_{v}_100m").add_prefix(f"w{v[0]}_")
                       for v in ("speed", "direction")], axis=1).rename_axis(columns=None)
        v = w[w.publishTime == tau].set_index("startTime").generation
        rows.append(X.assign(issue_time=tau + PUB_LAG, run=f.run.iloc[0], published=f.published.iloc[0], windfor_pub=tau,
                             windfor=v.reindex(X.index), y=y.reindex(X.index), y_metered=y_met.reindex(X.index)))
    df = pd.concat(rows).rename_axis("valid_time").reset_index()
    doy = 2 * np.pi * df.valid_time.dt.dayofyear / 365.25
    df = df.assign(cap=capacity.capacity(pd.DatetimeIndex(df.issue_time)).to_numpy(),
                   lead_h=(df.valid_time - df.issue_time).dt.total_seconds() / 3600, hour=df.valid_time.dt.hour,
                   doy_sin=np.sin(doy), doy_cos=np.cos(doy))
    check(df)
    return df

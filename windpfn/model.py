"""TabPFN-3.5 day-ahead forecast: physics features, capacity-factor target, point-in-time context."""
import numpy as np
import pandas as pd
from tabpfn_client import TabPFNRegressor
from . import baselines
from .dataset import WD, WS

EMBARGO = pd.Timedelta(days=1)  # archived BOAV cannot show when curtailment was known; train only on settled hours
TIME = ["cap", "lead_h", "hour", "doy_sin", "doy_cos"]


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Per-point power-curve capacity factor and wind direction, plus calendar and capacity."""
    wd = np.radians(d[WD])
    return pd.concat([d[WS].apply(np.interp, xp=baselines.PC_V, fp=baselines.PC_P).add_prefix("pc_"),
                      np.sin(wd).add_prefix("sin_"), np.cos(wd).add_prefix("cos_"), d[TIME]], axis=1)


def fit_predict(X: pd.DataFrame, d: pd.DataFrame, train, test) -> pd.Series:
    m = TabPFNRegressor.create_default_for_version("v3.5").fit(X[train], (d.y / d.cap)[train])
    return pd.Series(m.predict(X[test], output_type="median") * d.cap[test].to_numpy(), d.index[test])


def backtest(X: pd.DataFrame, d: pd.DataFrame, start, end=None, refit=True) -> pd.Series:
    """Forecast [start, end]. Context: hours settled by the first issue (less EMBARGO); refit monthly or frozen."""
    t = d.index.to_series()[start:end]
    blocks = [g.index for _, g in t.groupby(t.dt.strftime("%Y-%m"))] if refit else [t.index]
    return pd.concat(fit_predict(X, d, d.index <= d.issue_time.loc[b].min() - EMBARGO, d.index.isin(b)) for b in blocks)


def score(err: pd.DataFrame, ref="windfor", n=2000, seed=0) -> pd.DataFrame:
    """MAE, RMSE, bias (MW); ΔMAE vs `ref` with a 95% day-block bootstrap interval."""
    a = err.abs().groupby(err.index.floor("D")).mean()
    diff = a.sub(a[ref], axis=0)
    boot = diff.to_numpy()[np.random.default_rng(seed).integers(0, len(a), (n, len(a)))].mean(1)
    return pd.DataFrame({"MAE": err.abs().mean(), "RMSE": err.pow(2).mean() ** .5, "bias": err.mean(), "ΔMAE": diff.mean(),
                         "lo": np.percentile(boot, 2.5, 0), "hi": np.percentile(boot, 97.5, 0)}).round(0)

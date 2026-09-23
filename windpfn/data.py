"""Elexon BMRS loaders: WINDFOR vintages, FUELHH outturn, BOAV curtailment, prices, issue times."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .cache import get_json

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
B = f"{ELEXON}/datasets"
UTC = lambda s: pd.to_datetime(s, utc=True)
TODAY = lambda: pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _months(start, end):
    return pd.date_range(pd.Timestamp(start).to_period("M").start_time, end, freq="MS")


def windfor(start="2016-01-01", end=None):
    end = pd.Timestamp(end) if end else TODAY() + pd.Timedelta(days=1)
    d = pd.concat(pd.DataFrame(get_json("windfor", f"{B}/WINDFOR/stream", dict(
        publishDateTimeFrom=f"{a:%Y-%m-%dT%H:%MZ}", publishDateTimeTo=f"{a + pd.offsets.MonthBegin():%Y-%m-%dT%H:%MZ}")))
        for a in _months(start, end)).drop_duplicates()
    return d.assign(publishTime=UTC(d.publishTime), startTime=UTC(d.startTime))


def fuelhh(start="2016-01-01", end=None):
    end = pd.Timestamp(end) if end else TODAY()
    d = pd.concat(pd.DataFrame(get_json("fuelhh", f"{B}/FUELHH/stream", dict(
        settlementDateFrom=f"{a:%Y-%m-%d}", settlementDateTo=f"{min(a + pd.offsets.MonthEnd(0), end):%Y-%m-%d}",
        fuelType="WIND"))) for a in _months(start, end))
    # key on UTC startTime: API (settlementDate, SP) labels are wrong for SP48/clock days until 2022-07
    d = d.assign(startTime=UTC(d.startTime), publishTime=UTC(d.publishTime))
    return d.sort_values("publishTime").drop_duplicates("startTime", keep="last").set_index("startTime").generation.sort_index()


def curtailment(start, end, units):
    """MW removed from `units` by accepted BM bids net of offers (BOAV), per half-hour startTime.
    Zero where nothing was accepted. Live, BOAV lands ~1 h after the half-hour; archived rows carry
    createdDateTime = startTime + ~25 h, so backtests treat it as target-only.
    Misses self-curtailment via PNs at negative prices (not a BM action)."""
    def day(t):
        v = pd.concat(pd.DataFrame(get_json("boav", f"{ELEXON}/balancing/settlement/acceptance/volumes/all/{bo}/{t:%Y-%m-%d}",
                                            refresh=t >= TODAY() - pd.Timedelta(days=7))["data"]) for bo in ("bid", "offer"))
        return v[v.bmUnit.isin(units)] if len(v) else None
    with ThreadPoolExecutor(8) as ex:
        d = pd.concat(ex.map(day, pd.date_range(start, end or TODAY())))
    c = -2 * d.groupby(UTC(d.startTime)).totalVolumeAccepted.sum()
    return c.reindex(pd.date_range(UTC(start), c.index.max(), freq="30min"), fill_value=0.0).rename("curtailment")


def prices(start, end=None):
    """Day-ahead market index price (APX, £/MWh) per half-hour startTime."""
    d = pd.concat(pd.DataFrame(get_json("mid", f"{B}/MID/stream", {
        "from": f"{a:%Y-%m-%d}", "to": f"{a + pd.Timedelta(days=7):%Y-%m-%d}", "dataProviders": "APXMIDP"}))
        for a in pd.date_range(start, end or TODAY(), freq="7D"))
    return d.assign(startTime=UTC(d.startTime)).drop_duplicates("startTime").set_index("startTime").price.sort_index()


def issues(start="2024-03-16", end=None, cutoff="08:30"):
    """Delivery day D (UTC) -> publishTime of the latest WINDFOR vintage <= D-1 cutoff."""
    t = pd.Series(windfor(start, end).publishTime.unique()).sort_values()
    days = pd.date_range(UTC(start), t.max().floor("D") + pd.Timedelta(days=1), freq="D")
    cut = pd.DataFrame({"t": days - pd.Timedelta(days=1) + pd.Timedelta(cutoff + ":00"), "D": days})
    m = pd.merge_asof(cut, pd.DataFrame({"t": t, "issue": t}), on="t")
    m = m.set_index("D").issue.dropna()
    return m[:UTC(end)] if end else m


def dayahead_vintage(w, cutoff="08:30"):
    """Latest vintage published <= D-1 cutoff (UTC), future hours of D only."""
    w = w[w.startTime > w.publishTime]
    D = w.startTime.dt.floor("D")
    w = w[w.publishTime <= D - pd.Timedelta(days=1) + pd.Timedelta(cutoff + ":00")]
    return w.sort_values("publishTime").drop_duplicates("startTime", keep="last").set_index("startTime")

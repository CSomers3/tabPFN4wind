"""Point-in-time wind capacity: BMU nameplate x first metered output (weekly B1610 samples)."""
import pandas as pd
from .cache import get_json
from .data import ELEXON as API
LAG = pd.Timedelta(days=7)  # B1610 published ~5 days after delivery


def units() -> pd.Series:
    r = pd.DataFrame(get_json("bmunits", f"{API}/reference/bmunits/all"))
    r = r[(r.fuelType == "WIND") & r.elexonBmUnit.str[:2].isin(["T_", "E_"])]
    return r.drop_duplicates("elexonBmUnit").set_index("elexonBmUnit").generationCapacity.astype(float)


def first_output(start="2023-06-07", end=None) -> pd.Series:
    """BMU -> first weekly sample (Wed 11:00Z) with positive metered output."""
    cap, seen = units(), {}
    for t in pd.date_range(start, end or pd.Timestamp.now(), freq="7D"):
        b = pd.DataFrame(get_json("b1610", f"{API}/datasets/B1610/stream",
                                  {"from": f"{t:%Y-%m-%d}T11:00Z", "to": f"{t:%Y-%m-%d}T11:05Z"}))
        if b.empty:
            continue
        for u in b[b.bmUnit.isin(cap.index) & (b.quantity > 0)].bmUnit:
            seen.setdefault(u, pd.Timestamp(t, tz="UTC"))
    return pd.Series(seen, name="first_output")


def capacity(at: pd.DatetimeIndex, first: pd.Series | None = None) -> pd.Series:
    """MW of wind BMUs whose first output was known (published) by each time in `at`."""
    first = first_output() if first is None else first
    live = (first + LAG).sort_values()
    cum = units()[live.index].cumsum().to_numpy()
    i = live.searchsorted(at, side="right")
    return pd.Series([cum[k - 1] if k else 0.0 for k in i], at, name="capacity")

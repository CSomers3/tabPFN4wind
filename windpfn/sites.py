"""Wind farm sites (REPD) and capacity-weighted clusters used as NWP points."""
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
from .cache import DATA

REPD = DATA / "raw/repd/REPD_Q2_2026.csv"
REPD_URL = "https://assets.publishing.service.gov.uk/media/6a6cbdc00c36759b5ccaa305/REPD_Publication_Q2_2026.csv"
GENERATING = {"Dogger Bank A & B": "2023-11-01", "Sofia": "2026-04-22"}  # "Under Construction" but metering (B1610)
MIN_MW, K = 10, {"Wind Offshore": 10, "Wind Onshore": 10}
_to_ll = Transformer.from_crs(27700, 4326, always_xy=True).transform


def farms() -> pd.DataFrame:
    d = pd.read_csv(REPD, encoding="latin-1")
    st, tech = d["Development Status (short)"], d["Technology Type"]
    gen = d["Site Name"].str.startswith(tuple(GENERATING), na=False)
    d = d[((st == "Operational") & tech.str.startswith("Wind", na=False) | gen) & (d.Country != "Northern Ireland")]
    d = d.rename(columns={"Site Name": "name", "Installed Capacity (MWelec)": "mw", "X-coordinate": "x",
                          "Y-coordinate": "y", "Operational": "operational", "Technology Type": "tech"})
    d = d[["name", "tech", "mw", "x", "y", "operational"]].copy()
    d["name"] = d.name.str.strip(" \xa0")
    d["mw"], d["x"], d["y"] = (pd.to_numeric(d[c], errors="coerce") for c in ["mw", "x", "y"])
    d["operational"] = pd.to_datetime(d.operational, dayfirst=True, errors="coerce").dt.tz_localize("UTC")
    for k, t in GENERATING.items():
        d.loc[d.name.str.startswith(k), "operational"] = pd.Timestamp(t, tz="UTC")
    return d.dropna(subset=["mw", "x", "y"]).query("mw >= @MIN_MW").reset_index(drop=True)


def clusters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Farms labelled with a `point`, and capacity-weighted centroids per point."""
    f = farms()
    for j, (t, k) in enumerate(K.items()):
        m = f.tech == t
        lab = KMeans(k, n_init=20, random_state=0).fit(f[m][["x", "y"]], sample_weight=f.mw[m]).labels_
        f.loc[m, "point"] = [f"{'off' if j == 0 else 'on'}_{10 * j + i:02d}" for i in lab]
    wm = lambda g, c: (g[c] * g.mw).sum() / g.mw.sum()
    c = f.groupby("point").apply(lambda g: pd.Series({"x": wm(g, "x"), "y": wm(g, "y"), "mw": g.mw.sum(), "n": len(g),
                                                       "top": g.nlargest(1, "mw").name.iloc[0]}), include_groups=False)
    c["lon"], c["lat"] = _to_ll(c.x.values, c.y.values)
    return f, c.round({"lat": 3, "lon": 3, "mw": 0})


def write_points(path=DATA / "points.csv"):
    clusters()[1].reset_index()[["point", "lat", "lon", "mw", "n", "top"]].to_csv(path, index=False)
    return path

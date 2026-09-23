"""Publication figures (figures4papers house style). `d` is the table from dataset.build + baselines.add."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from .baselines import SPLIT
from .cache import get_json
from .sites import _to_ll

C = dict(y="#272727", met="#A5A5A5", windfor="#B64342", pc="#0F4D92", blend="#8BCF8B", light="#CFCECE")
NAME = dict(windfor="WINDFOR (NESO)", powercurve="power curve (ECMWF)", blend="50/50 blend")
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 2,
                     "xtick.major.width": 2, "ytick.major.width": 2, "xtick.major.size": 6, "ytick.major.size": 6,
                     "lines.linewidth": 2.5, "legend.frameon": False, "legend.fontsize": 13,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300})
GW = lambda ax, axis="y": getattr(ax, f"{axis}axis").set_major_formatter(lambda v, _: f"{v / 1e3:g}")
MAE = lambda s, m: (s[m] - s.y).abs().mean()


def _panels(ax, x=-0.14):
    for a, l in zip(np.ravel(ax), "abcdef"):
        a.text(x, 1.04, l, transform=a.transAxes, fontsize=19, fontweight="bold", va="bottom")


def timeline(d):
    """When everything happens, relative to delivery day D 00:00 UTC, for a summer and a winter day."""
    g = d.groupby(d.index.normalize()).first()
    h = lambda t: (t - g.index).dt.total_seconds() / 3600
    g = g.assign(run=h(g.run), pub=h(g.published), wf=h(g.windfor_pub), iss=h(g.issue_time))
    hh = lambda x: f"{x % 24 // 1:02.0f}:{x % 1 * 60:02.0f}"
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
    for ax, (name, s) in zip(axs, [("summer (BST)", g[g.iss > -15.5]), ("winter (GMT)", g[g.iss < -15.5])]):
        r = s.run.mode()[0]
        pub, wf, iss = s[s.run == r].pub.median(), s.wf.median(), s.iss.median()
        ax.plot([r, pub], [2, 2], c=C["pc"], lw=4, solid_capstyle="butt")
        ax.plot(r, 2, "o", ms=11, mfc="white", mec=C["pc"], mew=2.5)
        ax.plot(pub, 2, "o", ms=11, c=C["pc"], mec="black")
        ax.annotate(f"ECMWF {r % 24:02.0f}Z run\npublic {hh(pub)}", (r, 2), xytext=(-6, 11), textcoords="offset points", fontsize=13, c=C["pc"])
        ax.plot(wf, 1, "D", ms=10, c=C["windfor"], mec="black")
        ax.annotate(f"WINDFOR vintage {hh(wf)}", (wf, 1), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=13, c=C["windfor"])
        ax.barh(0, 24, left=0, height=.45, color=C["light"], ec="black", lw=1.5)
        ax.text(12, 0, f"24 hourly valid times, lead {-iss:.0f}–{23 - iss:.0f} h", ha="center", va="center", fontsize=13)
        ax.axvline(iss, c="black", ls="--", lw=1.5)
        ax.text(iss + .5, 1, f"we issue {hh(iss)}", va="center", fontsize=13, fontweight="bold")
        ax.set(yticks=[2, 1, 0], yticklabels=["weather", "benchmark", "delivery"], ylim=(-.6, 2.9), xlim=(-31, 24.5))
        runs = ", ".join(f"{k:02.0f}Z {v:.0%}" for k, v in s.run.mod(24).value_counts(normalize=True).items() if v > .005)
        ax.set_title(f"{name}, {len(s)} days. ECMWF run used: {runs}", loc="left", fontsize=14)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ticks = np.arange(-30, 25, 6)
    axs[1].set_xticks(ticks, [f"{t % 24:02.0f}:00" + (f"\n{'D−1 D D+1'.split()[t // 24 + 1]}" if t % 24 == 0 else "") for t in ticks])
    axs[1].set_xlabel("UTC")
    fig.tight_layout(pad=1)
    return fig


def coast(ax):
    g = get_json("naturalearth", "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson")
    for f in g["features"]:
        if f["properties"]["ADM0_A3"] in ("GBR", "IRL"):
            for poly in f["geometry"]["coordinates"]:
                ax.fill(*np.array(poly[0]).T, fc="#F4F4F4", ec="#767676", lw=.8, zorder=0)


def sites_map(farms, pts):
    fig, ax = plt.subplots(figsize=(5.5, 7))
    coast(ax)
    ax.scatter(*_to_ll(farms.x.values, farms.y.values), s=farms.mw / 8, c="#9A9A9A", lw=0, alpha=.7, label="wind farm (REPD)")
    off = pts.index.str.startswith("off")
    for m, cl, lab in [(off, C["pc"], "offshore point"), (~off, C["blend"], "onshore point")]:
        ax.scatter(pts.lon[m], pts.lat[m], s=pts.mw[m] / 8, c=cl, ec="black", lw=1, label=lab)
    ax.set(aspect=1.7, xlim=(-8.5, 3), ylim=(49.8, 59.5), xticks=[], yticks=[])
    ax.spines[["left", "bottom"]].set_visible(False)
    lg = ax.legend(loc="upper right", fontsize=13)
    for hd in lg.legend_handles: hd.set_sizes([80])
    fig.tight_layout(pad=.5)
    return fig


def target(d):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    b = (d.windfor // 1000) * 1000 + 500
    for col, cl, lab in [("y_metered", C["met"], "metered"), ("y", C["y"], "metered + curtailed")]:
        g = d[b.map(b.value_counts()) >= 30].groupby(b)[col]
        ax[0].fill_between(g.mean().index, g.quantile(.1), g.quantile(.9), color=cl, alpha=.2, lw=0)
        ax[0].plot(g.mean(), c=cl, label=lab)
    ax[0].plot([0, 2e4], [0, 2e4], c=C["windfor"], lw=1.5, ls="--", label="y = x")
    ax[0].set(xlabel="WINDFOR (GW)", ylabel="outturn (GW)", xlim=(0, 2e4), ylim=(0, 2e4), aspect=1)
    GW(ax[0]); GW(ax[0], "x"); ax[0].legend(loc="upper left")
    m = d.resample("MS")[["y_metered", "curtailed"]].mean()
    x = np.arange(len(m))
    ax[1].bar(x, m.y_metered, color=C["met"], ec="black", lw=1.5, label="metered")
    ax[1].bar(x, m.curtailed, bottom=m.y_metered, color=C["windfor"], ec="black", lw=1.5, label="curtailed")
    for i, (v, c) in enumerate(zip(m.sum(1), m.curtailed / m.sum(1))):
        ax[1].text(i, v + 150, f"{c:.0%}", ha="center", fontsize=13)
    ax[1].set_xticks(x, m.index.strftime("%b")); GW(ax[1])
    ax[1].set(ylabel="mean available wind (GW)"); ax[1].legend(loc="upper left")
    _panels(ax); fig.tight_layout(pad=1)
    return fig


def week(d, a="2024-11-01", b="2024-11-14"):
    s = d[a:b]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.y, c=C["y"], label="target")
    ax.plot(s.windfor, c=C["windfor"], lw=2, label=NAME["windfor"])
    ax.plot(s.powercurve, c=C["pc"], lw=2, label=NAME["powercurve"])
    GW(ax); ax.set(ylabel="GB wind (GW)", ylim=(0, None), xlim=(s.index[0], s.index[-1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15))
    fig.tight_layout(pad=1)
    return fig


def benchmarks(d):
    s = d[d.index >= SPLIT]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    cols = [C["windfor"], C["pc"], C["blend"]]
    bars = ax[0].bar(range(3), [MAE(s, m) for m in NAME], color=cols, ec="black", lw=1.5, width=.7)
    for bb in bars:
        ax[0].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 10, f"{bb.get_height():.0f}", ha="center", va="bottom", fontsize=13)
    ax[0].set_xticks(range(3), ["WINDFOR", "power\ncurve", "blend"]); ax[0].set(ylabel="MAE, Aug–Dec (MW)", ylim=(0, 1150))
    q = pd.qcut((s.windfor + s.powercurve) / 2, 4, labels=["low", "mid", "high", "very high"])
    g = s.groupby(q, observed=True).apply(lambda g: pd.Series({m: MAE(g, m) for m in NAME}))
    x, w = np.arange(4), .27
    for i, (m, cl) in enumerate(zip(NAME, cols)):
        ax[1].bar(x + (i - 1) * w, g[m], w, color=cl, ec="black", lw=1.5, label=NAME[m])
    ax[1].set_xticks(x, g.index); ax[1].set(xlabel="forecast level (quartile, known at issue)", ylabel="MAE (MW)")
    ax[1].legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=3)
    _panels(ax); fig.tight_layout(pad=1)
    return fig


def errors(d):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    q, s = pd.qcut(d.powercurve, 10), d[d.index >= SPLIT]
    for m, cl in [("windfor", C["windfor"]), ("powercurve", C["pc"])]:
        e = d[m] - d.y
        ax[0].plot([i.mid for i in q.cat.categories], e.groupby(q, observed=True).std(), "o-", c=cl, label=NAME[m], ms=7, mec="black")
        ax[1].plot((s[m] - s.y).abs().groupby(s.lead_h).mean(), c=cl)
        ax[2].plot(range(73), [e.autocorr(k) for k in range(73)], c=cl)
    GW(ax[0], "x"); ax[0].set(xlabel="forecast level (GW)", ylabel="error s.d. (MW)", ylim=(0, None)); ax[0].legend(loc="lower right")
    ax[1].set(xlabel="lead time (h)", ylabel="MAE, Aug–Dec (MW)")
    ax[2].axhline(0, c="black", lw=1); ax[2].set(xlabel="lag (h)", ylabel="error autocorrelation", xticks=range(0, 73, 24))
    _panels(ax); fig.tight_layout(pad=1)
    return fig


def monthly(err):
    """Monthly MAE per forecast; runs with a frozen context dashed."""
    mae = err.abs().resample("MS").mean()
    mae.index = mae.index.strftime("%b %y")
    col = {"windfor": C["windfor"], "blend": C["met"], "tabpfn": C["pc"], "tabpfn+windfor": C["blend"]}
    fig, ax = plt.subplots(figsize=(13, 4.3))
    for k in mae:
        ax.plot(mae.index, mae[k], "o--" if "frozen" in k else "o-", c=col[k.split()[0]], ms=7, mec="black", label=k)
    ax.set(ylabel="MAE (MW)", ylim=(0, None)); ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    fig.tight_layout(pad=1)
    return fig


def save_all(d, farms, pts, out="figures"):
    out = Path(out); out.mkdir(exist_ok=True)
    figs = dict(timeline=timeline(d), sites=sites_map(farms, pts), target=target(d), week=week(d),
                benchmarks=benchmarks(d), errors=errors(d))
    for k, f in figs.items():
        for ext in ["png", "pdf"]:
            f.savefig(out / f"{k}.{ext}", bbox_inches="tight", pad_inches=.05)
        plt.close(f)
    return list(figs)

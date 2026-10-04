"""The palette and matplotlib style shared by the notebooks.

`use` is the dark notebook style: amber is TabPFN, red is NESO and nothing else, grey is outturn.
`paper` is the README figures', copied from the live page: blue is TabPFN, raspberry is NESO and
anything built only on it, ink is outturn, in the bundled Inter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

PALETTE = {
    "bg": "#0A0E13",
    "panel": "#0F151C",
    "ink": "#E8EDF2",
    "ink2": "#9AA7B4",
    "muted": "#76838F",
    "rule": "#1D2733",
    "rule2": "#141C26",
    "forecast": "#F2A93B",
    "incumbent": "#E5484D",
    "outturn": "#C9D4DE",
}

PAPER = {  # notebooks/explorer.py's palette, so the README figures match the live page
    "bg": "#FFFFFF",
    "past": "#F5F7FA",
    "ink": "#0F172A",
    "ink2": "#475569",
    "muted": "#64748B",
    "faint": "#94A3B8",
    "rule": "#CBD5E1",
    "rule2": "#EDF1F5",
    "outturn": "#0F172A",
    "incumbent": "#C2255C",
    "forecast": "#3E7AC4",
    "median": "#1F5FAD",
    "fan": ["#DCE7F4", "#C1D4EC", "#A6C2E4", "#8BAFDC"],
    "gap": "#E4EDF9",
}

FAN = ((10, 90), (20, 80), (30, 70), (40, 60))  # central intervals, outermost first, shaded deeper inward
LABELLED = (90, 70, 50, 30, 10)

PX = 0.75  # CSS px -> pt

for _font in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(_font)


def use() -> None:
    """Set matplotlib to the page's style."""
    p = PALETTE
    mpl.rcParams.update({
        "figure.facecolor": p["bg"],
        "axes.facecolor": p["bg"],
        "savefig.facecolor": p["bg"],
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.dpi": 110,
        "font.family": "monospace",
        "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"],
        "font.size": 11 * PX,
        "text.color": p["ink2"],
        "axes.labelcolor": p["ink2"],
        "axes.labelsize": 11 * PX,
        "axes.titlesize": 11 * PX,
        "axes.edgecolor": p["rule"],
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": p["rule2"],
        "grid.linewidth": 0.75,
        "grid.linestyle": (0, (3, 3)),
        "axes.axisbelow": True,
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "xtick.labelcolor": p["muted"],
        "ytick.labelcolor": p["muted"],
        "xtick.labelsize": 9 * PX,
        "ytick.labelsize": 9 * PX,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "legend.frameon": False,
        "legend.fontsize": 9 * PX,
        "legend.labelcolor": p["ink2"],
        "legend.handlelength": 1.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.prop_cycle": mpl.cycler(color=[p["forecast"], p["incumbent"], p["outturn"], p["muted"]]),
    })


def paper():
    """A context for the README figures, in the live page's type and colours."""
    p = PAPER
    return mpl.rc_context({
        "figure.facecolor": p["bg"],
        "axes.facecolor": p["bg"],
        "savefig.facecolor": p["bg"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "Inter",
        "font.size": 7,
        "text.color": p["ink"],
        "axes.labelcolor": p["muted"],
        "axes.labelsize": 7,
        "axes.spines.left": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": p["rule2"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.labelcolor": p["faint"],
        "ytick.labelcolor": p["faint"],
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.labelcolor": p["ink2"],
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.8,
        "legend.borderaxespad": 0,
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


class Fan:
    """Legend handle for the predictive distribution: the page's nested central intervals and median."""


class _FanHandler:
    def legend_artist(self, legend, handle, fontsize, box):
        middle, half = box.ydescent + box.height / 2, box.height * 0.75
        for k, colour in enumerate(PAPER["fan"]):
            height = half * (1 - k / len(FAN))
            box.add_artist(mpl.patches.Rectangle(
                (box.xdescent, middle - height), box.width, 2 * height, facecolor=colour, lw=0,
            ))
        box.add_artist(mpl.lines.Line2D(
            [box.xdescent, box.xdescent + box.width], [middle, middle], color=PAPER["median"], lw=1.4,
        ))
        return box


def header(ax, unit: str, handles=(), labels=(), y: float = 1.04) -> None:
    """The page's top row, at height `y` in axes units: the unit on the left, the legend on the right."""
    ax.text(0, y, unit, transform=ax.transAxes, color=PAPER["muted"], va="bottom")
    if handles:
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1, y - 0.02), ncol=len(handles),
                                 handler_map={Fan: _FanHandler()})


def line(colour: str, width: float = 1.1):
    return mpl.lines.Line2D([], [], color=colour, lw=width)


def fan_chart(forecast: pd.DataFrame, incumbent: pd.Series, observed: pd.Series, now=None,
              label: str = "TabPFN-3.5", path: Path | None = None):
    """The live page's chart, in GW: outturn under the forecast's fan and median, and NESO's forecast.

    `forecast` holds q10..q90 on an hourly index. With `now`, the hours before it are shaded.
    With `path`, the chart is also saved there.
    """
    p = PAPER
    hours = forecast.index
    span = hours[-1] - hours[0]
    ceiling = max(10, np.ceil(np.nanmax([forecast.q90.max(), incumbent.max(), observed.max()]) / 5) * 5)

    with paper():
        figure, ax = plt.subplots(figsize=(6.4, 2.6))
        if now is not None:
            ax.axvspan(hours[0], now, color=p["past"], lw=0, zorder=0)
        for (lower, upper), colour in zip(FAN, p["fan"]):
            ax.fill_between(hours, forecast[f"q{lower}"], forecast[f"q{upper}"], color=colour, lw=0)
        ax.plot(forecast.q50, color=p["median"], lw=1.4)
        ax.plot(incumbent, color=p["incumbent"])
        ax.plot(observed, color=p["outturn"], lw=1.4)
        ax.axhline(0, color=p["rule"], lw=0.6)
        if now is not None:
            ax.plot(observed.index[-1], observed.iloc[-1], "o", ms=4, color=p["outturn"], mec=p["bg"], mew=1.2)
            ax.vlines(now, 0, ceiling * 1.03, color=p["faint"], lw=0.6, ls=(0, (4, 3)), clip_on=False)
            ax.text(now, ceiling * 1.05, "Now", ha="center", va="bottom", color=p["muted"])

        last = forecast.q50.last_valid_index()
        edges = [forecast.at[last, f"q{q}"] for q in LABELLED]
        ys, middle, gap = list(edges), len(LABELLED) // 2, ceiling * 0.055
        for k in range(middle - 1, -1, -1):
            ys[k] = max(ys[k], ys[k + 1] + gap)
        for k in range(middle + 1, len(ys)):
            ys[k] = min(ys[k], ys[k - 1] - gap)
        for q, edge, y in zip(LABELLED, edges, ys):
            ax.annotate(f"p{q}", (last, edge), xytext=(last + span * 0.031, y), va="center",
                        fontsize=6.5, color=p["ink"] if q == 50 else p["muted"],
                        fontweight="semibold" if q == 50 else "normal", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": p["rule"], "lw": 0.6, "shrinkA": 0, "shrinkB": 1})

        ax.set(xlim=(hours[0], hours[-1]), ylim=(0, ceiling), yticks=np.arange(0, ceiling + 1, 5))
        for day in hours[hours.hour == 0][1:]:
            ax.axvline(day, color=p["rule2"], lw=0.6, zorder=0.5)
        ticks = hours[hours.hour % (6 if span <= pd.Timedelta(days=3) else 24) == 0]
        ax.set_xticks(ticks, [f"{t:%a %d %b}" if t.hour == 0 else f"{t:%H:%M}" for t in ticks])
        for t, tick in zip(ticks, ax.get_xticklabels()):
            if t.hour == 0:
                tick.set(color=p["ink"], fontweight="semibold")

        header(ax, "Wind generation, GW · UTC", [line(p["outturn"], 1.4), Fan(), line(p["incumbent"])],
               ["Outturn", label, "NESO"], y=1.12)
        if path:
            figure.savefig(path)
    return figure

"""The palette and matplotlib style shared by the notebooks.

`use` is the dark notebook style: amber is TabPFN, red is NESO and nothing else, grey is outturn.
`paper` is the README figures', copied from the live page: blue is TabPFN, raspberry is NESO and
anything built only on it, ink is outturn, in the bundled Inter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
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
}

FAN = mpl.colors.LinearSegmentedColormap.from_list("fan", ["#E4EDF9", PAPER["forecast"]])  # p0/p100 -> p50

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
    """Legend handle for the predictive distribution: the page's blue swatch, darkest at the median."""


class _FanHandler:
    def legend_artist(self, legend, handle, fontsize, box):
        middle, half = box.height / 2, box.height * 0.75
        for depth in np.linspace(0.05, 1, 12):
            box.add_artist(mpl.patches.Rectangle(
                (box.xdescent, box.ydescent + middle - half * (1.05 - depth)), box.width,
                2 * half * (1.05 - depth), facecolor=FAN(depth), lw=0,
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

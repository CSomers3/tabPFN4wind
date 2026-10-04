import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 · Held-out test: TabPFN-3.5 vs NESO, 2025–2026

    The model from `02`, scored once on 2025-01-01 → 2026-09-15 under the protocol frozen
    at `8e4386d`, on ECMWF IFS weather (`ifs`, committed at `d7df551`). The package then
    moved to the open AIFS archive on dynamical.org, and TabPFN was rerun unchanged on it
    (`aifs`). Both were produced by `windpfn-backtest tabpfn`; this notebook only reads
    them, so it needs no network and no token.

    Monthly runs refit on every hour settled a day before issue; `frozen` runs keep the
    first month's context throughout. `blend` is the fixed 50/50 WINDFOR / power-curve
    benchmark, the bar for `tabpfn+windfor`.

    Produces the two held-out README figures, `miss.png` and `calibration.png`.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    from windpfn import evaluation, sources, style

    style.use()
    WIDTH = 7.5
    TABPFN = {"tabpfn+windfor ifs": "TabPFN-3.5 · IFS", "tabpfn+windfor aifs": "TabPFN-3.5 · AIFS"}
    return TABPFN, WIDTH, evaluation, pd, plt, sources, style


@app.cell
def _(evaluation):
    forecasts = evaluation.load_forecasts()
    _frozen = [f"tabpfn frozen {weather}" for weather in evaluation.WEATHER]
    deciles, errors, scores = evaluation.scorecard(forecasts, [*evaluation.PROBABILISTIC, *_frozen])
    runs = ["windfor", "blend", *[f"{run} {w}" for run in ("tabpfn", "tabpfn+windfor") for w in evaluation.WEATHER]]
    scores.loc[runs, ["MAE", "RMSE", "bias", "ΔMAE", "lo", "hi"]]
    return deciles, errors, forecasts, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By month

    IFS runs dashed, AIFS runs solid.
    """)
    return


@app.cell
def _(WIDTH, errors, plt, runs, style):
    _p = style.PALETTE
    colours = {"windfor": _p["incumbent"], "blend": _p["muted"], "tabpfn": _p["ink2"], "tabpfn+windfor": _p["forecast"]}
    monthly = errors[runs].abs().resample("MS").mean()

    figure, ax = plt.subplots(figsize=(WIDTH, 2.4))
    for run in runs:
        ax.plot(
            monthly.index, monthly[run], color=colours[run.split()[0]],
            ls="--" if run.endswith(" ifs") else "-", marker="o", ms=2.5, label=run,
        )
    ax.set(ylabel="MAE (MW)", ylim=(0, None))
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By lead time
    """)
    return


@app.cell
def _(errors, forecasts, pd, runs):
    lead = pd.cut(forecasts.lead_h, [15, 21, 27, 33, 40])
    errors[runs].abs().groupby(lead, observed=True).mean().round(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## README figures

    In the live page's type and colours (`style.paper`). First, cumulative absolute error over
    every held-out hour: NESO's WINDFOR as issued, and TabPFN-3.5's median after
    post-processing it, on AIFS.
    """)
    return


@app.cell
def _(TABPFN, forecasts, plt, sources, style):
    import matplotlib.dates as mdates

    PAPER_WIDTH = 6.4
    _p = style.PAPER
    BEST = "tabpfn+windfor aifs"
    windfor, tabpfn = ((forecasts.y - forecasts[_run]).abs().cumsum() / 1e6 for _run in ("windfor", f"{BEST}|q50"))
    end = windfor.index[-1]

    with style.paper():
        figure_miss, miss_ax = plt.subplots(figsize=(PAPER_WIDTH, 2.4))
        miss_ax.fill_between(windfor.index, tabpfn, windfor, color=_p["gap"], lw=0)
        miss_ax.plot(windfor, color=_p["incumbent"])
        miss_ax.plot(tabpfn, color=_p["forecast"])
        miss_ax.axhline(0, color=_p["rule"], lw=0.6)
        for _line in (windfor, tabpfn):
            miss_ax.annotate(f"{_line.iloc[-1]:.1f}", (end, _line.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                             va="center", color=_p["muted"])
        miss_ax.annotate(f"−{1 - tabpfn.iloc[-1] / windfor.iloc[-1]:.1%}", (end, (windfor.iloc[-1] + tabpfn.iloc[-1]) / 2),
                         xytext=(24, 0), textcoords="offset points", va="center", fontweight="semibold")

        miss_ax.set(xlim=(windfor.index[0], end), ylim=(0, 16), yticks=[0, 5, 10, 15])
        miss_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        miss_ax.xaxis.set_major_formatter(
            lambda value, _: mdates.num2date(value).strftime("%b %Y" if mdates.num2date(value).month == 1 else "%b")
        )
        miss_ax.figure.canvas.draw()
        for _tick in miss_ax.get_xticklabels():
            if _tick.get_text().startswith("Jan"):
                _tick.set(color=_p["ink"], fontweight="semibold", ha="left" if _tick.get_text().endswith("2025") else "center")
        miss_ax.axvline(mdates.datestr2num("2026-01-01"), color=_p["rule2"], lw=0.6)
        style.header(miss_ax, "Cumulative absolute error, TWh", [style.line(_p["incumbent"]), style.line(_p["forecast"])],
                     ["NESO", "TabPFN-3.5"])
        figure_miss.savefig(sources.ROOT / "notebooks" / "miss.png")
    figure_miss
    return (PAPER_WIDTH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calibration and sharpness of the central intervals the deciles give, p40–p60 to p10–p90,
    against the two reference forecasters from `04`. A forecast should sit on zero on the
    left; among those that do, narrower on the right is better.
    """)
    return


@app.cell
def _(PAPER_WIDTH, TABPFN, forecasts, plt, sources, style):
    _p = style.PAPER
    runs_shown = {
        "tabpfn+windfor ifs": ("TabPFN-3.5 · IFS", _p["fan"][2]),
        "tabpfn+windfor aifs": ("TabPFN-3.5 · AIFS", _p["median"]),
        "conformal": ("NESO + conformal", _p["incumbent"]),
        "lgbm+windfor": ("LightGBM · AIFS", _p["faint"]),
    }
    levels = [20, 40, 60, 80]

    with style.paper():
        figure_calibration, (coverage_ax, width_ax) = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.3), gridspec_kw={"wspace": 0.22})
        for _run, (_label, _colour) in runs_shown.items():
            _lower = [forecasts[f"{_run}|q{50 - _level // 2}"] for _level in levels]
            _upper = [forecasts[f"{_run}|q{50 + _level // 2}"] for _level in levels]
            _coverage = [100 * forecasts.y.between(_low, _high).mean() - _level for _low, _high, _level in zip(_lower, _upper, levels)]
            _width = [(_high - _low).mean() / 1e3 for _low, _high in zip(_lower, _upper)]
            for _axis, _values in ((coverage_ax, _coverage), (width_ax, _width)):
                _axis.plot(levels, _values, color=_colour, marker="o", ms=4, mec=_p["bg"], mew=1.2)
        coverage_ax.axhline(0, color=_p["rule"], lw=0.6)
        coverage_ax.set(ylim=(-15, 5), yticks=[-15, -10, -5, 0, 5])
        width_ax.set(ylim=(0, 3.5), yticks=[0, 1, 2, 3])
        for _axis in (coverage_ax, width_ax):
            _axis.set(xlim=(12, 88), xticks=levels)
            _axis.set_xticklabels([f"p{50 - _level // 2}–p{50 + _level // 2}" for _level in levels])
        style.header(coverage_ax, "Coverage minus nominal, percentage points")
        style.header(width_ax, "Mean interval width, GW")
        figure_calibration.legend([style.line(_colour) for _, _colour in runs_shown.values()],
                                  [_label for _label, _ in runs_shown.values()], loc="lower right",
                                  bbox_to_anchor=(width_ax.get_position().x1, 1.0), ncol=4)
        figure_calibration.savefig(sources.ROOT / "notebooks" / "calibration.png")
    figure_calibration
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worst days

    For each weather source, the eight days TabPFN lost worst to NESO, and the share of their
    hours inside its 10–90% band. Calibrated on average is not calibrated on bad days.
    """)
    return


@app.cell
def _(TABPFN, deciles, errors, forecasts, pd):
    daily = errors.abs().groupby(errors.index.floor("D")).mean()
    _worst = {}
    for _run in TABPFN:
        _q = deciles[_run]
        _in_band = forecasts.y.between(_q.q10, _q.q90).groupby(forecasts.index.floor("D")).mean()
        _lost = (daily[_run] - daily.windfor).nlargest(8)
        print(f"{_run}: inside 10–90% on its worst days, {_in_band[_lost.index].mean():.0%} of hours")
        _worst[_run] = _lost.round(0).to_frame("MAE lost vs NESO (MW)").join(_in_band.rename("in band").round(2))
    pd.concat(_worst)
    return


if __name__ == "__main__":
    app.run()

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

    Draws `assets/miss.png` and `assets/calibration.png`.
    """)
    return


@app.cell
def _():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    from windpfn import evaluation, sources, style

    style.use()
    TABPFN = {"tabpfn+windfor ifs": "TabPFN-3.5 · IFS", "tabpfn+windfor aifs": "TabPFN-3.5 · AIFS"}
    return TABPFN, evaluation, mdates, pd, plt, sources, style


@app.cell
def _(evaluation):
    forecasts = evaluation.load_forecasts()
    deciles, errors, scores = evaluation.scorecard(forecasts)
    runs = ["windfor", "blend", *[f"{run} {w}" for run in ("tabpfn", "tabpfn+windfor", "tabpfn+windfor frozen") for w in evaluation.WEATHER]]
    scores.loc[runs, ["MAE", "RMSE", "bias", "ΔMAE", "lo", "hi"]]
    return deciles, errors, forecasts, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By month
    """)
    return


@app.cell
def _(TABPFN, errors, mdates, plt, style):
    _p = style.PALETTE
    _shown = {"windfor": ("NESO", _p["incumbent"]), "tabpfn+windfor ifs": (TABPFN["tabpfn+windfor ifs"], _p["fan"][2]),
              "tabpfn+windfor aifs": (TABPFN["tabpfn+windfor aifs"], _p["median"])}
    _monthly = errors[list(_shown)].abs().resample("MS").mean()
    figure_monthly, ax = plt.subplots(figsize=(style.WIDTH, 2.2))
    for _run, (_label, _colour) in _shown.items():
        ax.plot(_monthly[_run], color=_colour, marker="o", ms=3, mec=_p["bg"], mew=1)
    ax.set(ylim=(0, None))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    style.header(ax, "Mean absolute error by month, MW", [style.line(c) for _, c in _shown.values()],
                 [label for label, _ in _shown.values()])
    figure_monthly
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

    First, cumulative absolute error over every held-out hour: NESO's WINDFOR as issued, and
    TabPFN-3.5's median after post-processing it, on AIFS.
    """)
    return


@app.cell
def _(forecasts, mdates, plt, sources, style):
    _p = style.PALETTE
    BEST = "tabpfn+windfor aifs"
    windfor, tabpfn = ((forecasts.y - forecasts[_run]).abs().cumsum() / 1e6 for _run in ("windfor", f"{BEST}|q50"))
    end = windfor.index[-1]

    figure_miss, miss_ax = plt.subplots(figsize=(style.WIDTH, 2.4))
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
    figure_miss.savefig(sources.ROOT / "assets" / "miss.png")
    figure_miss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calibration and sharpness of the central intervals the deciles give, nominal coverage 20%
    to 80%, on identical AIFS inputs: TabPFN-3.5, the tuned LightGBM and quantile regression
    forest from `04`, and the conformal reference. A forecast should sit on zero on the left;
    among those that do, narrower on the right is better.
    """)
    return


@app.cell
def _(forecasts, plt, sources, style):
    _p = style.PALETTE
    runs_shown = {
        "lightgbm": ("LightGBM", _p["faint"]),
        "qrf": ("Quantile forest", _p["faint"]),
        "conformal": ("NESO recalibrated + conformal", _p["incumbent"]),
        "tabpfn+windfor aifs": ("TabPFN-3.5", _p["median"]),
    }
    references = ("lightgbm", "qrf")
    levels = [20, 40, 60, 80]

    figure_calibration, (coverage_ax, width_ax) = plt.subplots(1, 2, figsize=(style.WIDTH, 2.3), gridspec_kw={"wspace": 0.22})
    for _run, (_label, _colour) in runs_shown.items():
        _lower = [forecasts[f"{_run}|q{50 - _level // 2}"] for _level in levels]
        _upper = [forecasts[f"{_run}|q{50 + _level // 2}"] for _level in levels]
        _coverage = [100 * forecasts.y.between(_low, _high).mean() - _level for _low, _high, _level in zip(_lower, _upper, levels)]
        _width = [(_high - _low).mean() / 1e3 for _low, _high in zip(_lower, _upper)]
        for _axis, _values in ((coverage_ax, _coverage), (width_ax, _width)):
            _axis.plot(levels, _values, color=_colour, marker="o", ms=4, mec=_p["bg"], mew=1.2,
                       lw=0.9 if _run in references else 1.4)
            if _run in references:
                _at, _side = (3, "right") if _axis is coverage_ax else (2, "right" if _run == "qrf" else "left")
                _up = _axis is width_ax and _run == "qrf"
                _axis.annotate(_label, (levels[_at], _values[_at]), xytext=(-3 if _side == "right" else 3, 7 if _up else -7),
                               textcoords="offset points", ha=_side, va="center", color=_p["muted"], fontsize=6)
    coverage_ax.axhline(0, color=_p["rule"], lw=0.6)
    coverage_ax.set(ylim=(-17, 10), yticks=[-15, -10, -5, 0, 5, 10])
    width_ax.set(ylim=(0, 4), yticks=[0, 1, 2, 3, 4])
    for _axis in (coverage_ax, width_ax):
        _axis.set(xlim=(12, 88), xticks=levels)
        _axis.set_xticklabels([f"{_level}%" for _level in levels])
    style.header(coverage_ax, "Coverage minus nominal, percentage points")
    style.header(width_ax, "Mean interval width, GW")
    for _axis in (coverage_ax, width_ax):
        _axis.set_xlabel("Nominal coverage of the central interval")
    _legend = {"tabpfn+windfor aifs": "TabPFN-3.5", "conformal": "NESO recalibrated + conformal", "lightgbm": "References"}
    figure_calibration.legend([style.line(runs_shown[_run][1], 1.4 if _run == "tabpfn+windfor aifs" else 0.9) for _run in _legend],
                              list(_legend.values()), loc="lower right",
                              bbox_to_anchor=(width_ax.get_position().x1, 1.0), ncol=4)
    figure_calibration.savefig(sources.ROOT / "assets" / "calibration.png")
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

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
    # 01 · Data, target and benchmark

    **Question.** Using ECMWF's open AIFS weather forecasts, can we predict tomorrow's hourly GB
    wind better than NESO's own forecast, WINDFOR, issued at the same moment?

    Window: 2024-04-02 → 2024-12-31. Anything fitted is fitted on Apr–Jul and scored on Aug–Dec;
    the 2025+ test window is not scored here. Needs the raw pulls (`windpfn-fetch`).
    """)
    return


@app.cell
def _():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.linear_model import LinearRegression

    from windpfn import dataset, evaluation, forecasting, sources, style, weather

    style.use()
    return (
        LinearRegression,
        LinearSegmentedColormap,
        dataset,
        evaluation,
        forecasting,
        mdates,
        np,
        pd,
        plt,
        sources,
        style,
        weather,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When the forecast is made

    NESO publishes WINDFOR eight times a day, hourly to two days ahead; its issue times shift by
    an hour in UTC with British Summer Time. Each delivery day D is forecast from the last vintage
    published by D−1 08:30 UTC, fifteen minutes after it, so both forecasts share an information
    cutoff. The weather must be public by then: dynamical.org keeps no publication time per run,
    and its update history shows each AIFS run landing 6h14m after initialisation, so a run is
    taken as public 6h30m after. `dataset.check` enforces every timing on every row.

    Median times relative to D 00:00 UTC:
    """)
    return


@app.cell
def _(dataset, forecasting):
    table = forecasting.add_baselines(dataset.build("2024-04-02", "2024-12-31"))
    scored = table[table.index >= forecasting.BASELINE_SPLIT]
    return scored, table


@app.cell
def _(pd, table):
    def _clock(delta: pd.Timedelta) -> str:
        day = "D−1" if delta < pd.Timedelta(0) else "D"
        return f"{day} {(pd.Timestamp(0) + delta):%H:%M}"

    _days = table.groupby(table.index.normalize()).first()
    _times = pd.DataFrame({
        "AIFS run": _days.run,
        "AIFS public": _days.published,
        "WINDFOR published": _days.windfor_pub,
        "issued": _days.issue_time,
    }).sub(_days.index, axis=0)
    _season = _days.issue_time.dt.tz_convert("Europe/London").dt.strftime("%Z").rename("clocks")
    _leads = table.lead_h.groupby(table.issue_time.dt.tz_convert("Europe/London").dt.strftime("%Z")).agg(["min", "max"])
    _summary = _times.groupby(_season).median().map(_clock)
    _summary["lead, h"] = [f"{low:.0f}–{high:.0f}" for low, high in _leads.loc[_summary.index].to_numpy()]
    _summary.assign(days=_season.value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data in

    | source | what | resolution | used as |
    |---|---|---|---|
    | ECMWF AIFS Single (dynamical.org) | 10 m wind at 20 points | 0.25°, 6-hourly → hourly | **features** |
    | Elexon BM units + B1610 | metered wind capacity | weekly | **feature** |
    | NESO WINDFOR | NESO's wind forecast | hourly | **benchmark**, and a feature |
    | Elexon FUELHH | metered wind output | half-hourly | **target**, part 1 |
    | Elexon BOAV | wind NESO paid to switch off | half-hourly | **target**, part 2 |

    The 20 points are capacity-weighted k-means clusters of GB wind farms of at least 10 MW in
    the Renewable Energy Planning Database: 10 offshore, 10 onshore. Capacity counts each BM unit
    from its first metered B1610 output, plus 7 days' publication lag, at its current nameplate.

    Other weather sources were rejected: Open-Meteo's historical forecasts stitch the first hours
    of each run into a near-analysis, its previous-runs archive leaks runs from after the cutoff,
    and ECMWF IFS open data comes only as global fields, about 324 GB to cover every run.
    """)
    return


@app.cell
def _(dataset, table):
    table[dataset.SPEEDS + dataset.COVARIATES + ["windfor", "y"]].iloc[[0, 12]].T.round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The weather columns, development and test

    The 20 power-curve columns TabPFN-3.5 reads, daily means, offshore then onshore points,
    each north to south; above them the target. Reads the committed `data/history/`.
    """)
    return


@app.cell
def _(LinearSegmentedColormap, dataset, evaluation, forecasting, mdates, np, pd, plt, sources, style, weather):
    _p = style.PALETTE
    _history = dataset.build(sources.ARCHIVE_START, evaluation.TEST_END, dataset.load(dataset.HISTORY))
    _points = weather.POINTS.assign(offshore=weather.POINTS.index.str.startswith("off"))
    _order = _points.sort_values(["offshore", "lat"], ascending=False).index
    _daily = forecasting.power_curve(_history).set_axis(_points.index, axis=1)[_order].resample("D").mean()
    _outturn = (_history.y / _history.cap).resample("D").mean()
    _test = pd.Timestamp(evaluation.TEST_START, tz="UTC")
    _offshore = _points.offshore.sum()

    figure_features, (_top, _heat) = plt.subplots(
        2, 1, figsize=(style.WIDTH, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.4}
    )
    _top.plot(_outturn, color=_p["outturn"], lw=0.7)
    _top.set(ylim=(0, 1), yticks=[0, 0.5, 1])
    _cmap = LinearSegmentedColormap.from_list("fan", [_p["bg"], *_p["fan"], _p["forecast"], _p["median"]])
    _mesh = _heat.pcolormesh(_daily.index, np.arange(len(_order)), _daily.T.to_numpy(), cmap=_cmap, vmin=0, vmax=1,
                             shading="nearest", rasterized=True)
    _heat.grid(False)
    _heat.axhline(_offshore - 0.5, color=_p["bg"], lw=1.5)
    _heat.set(ylim=(len(_order) - 0.5, -0.5))
    _heat.set_yticks([(_offshore - 1) / 2, _offshore + (len(_order) - _offshore - 1) / 2], ["Offshore", "Onshore"])
    _top.axvline(_test, color=_p["ink2"], lw=0.8)
    _heat.axvline(_test, color=_p["bg"], lw=1.5)
    _top.text(_test, 0.93, "  Held-out test →", va="center", color=_p["ink"], fontweight="semibold")
    _heat.set_xlim(_daily.index[0], _daily.index[-1])
    _heat.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    _heat.xaxis.set_major_formatter(
        lambda value, _: mdates.num2date(value).strftime("%b %Y" if mdates.num2date(value).month == 1 else "%b")
    )
    style.header(_top, "Outturn, capacity factor, daily mean")
    style.header(_heat, "Power-curve capacity factor at each weather point, north to south, daily mean")
    _bar = figure_features.colorbar(_mesh, cax=_heat.inset_axes([0.86, 1.05, 0.14, 0.045]), orientation="horizontal",
                                    ticks=[0, 0.5, 1])
    _bar.outline.set_visible(False)
    _bar.ax.tick_params(labelsize=6, pad=2)
    _bar.ax.xaxis.set_ticks_position("top")
    figure_features.savefig(sources.ROOT / "assets" / "features.png")
    figure_features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The target: what WINDFOR forecasts

    On windy days the grid between Scotland and England fills, and NESO pays Scottish farms to
    switch off, so metered output falls short of what the wind could have made. WINDFOR is an
    input to that decision and forecasts output before curtailment: against metered output it
    looks biased high, and with curtailment added back the two sit on the diagonal. The target
    `y` is therefore metered output plus balancing-mechanism curtailment.
    """)
    return


@app.cell
def _(plt, style, table):
    _p = style.PALETTE
    _level = (table.windfor // 1000) * 1000 + 500
    _binned = table[_level.map(_level.value_counts()) >= 30].groupby(_level)[["y_metered", "y"]].mean() / 1e3

    figure_target, ax = plt.subplots(figsize=(style.WIDTH / 2, style.WIDTH / 2))
    ax.plot([0, 20], [0, 20], color=_p["rule"], lw=0.8)
    ax.plot(_binned.index / 1e3, _binned.y_metered, color=_p["faint"], marker="o", ms=3, mec=_p["bg"], mew=1)
    ax.plot(_binned.index / 1e3, _binned.y, color=_p["outturn"], marker="o", ms=3, mec=_p["bg"], mew=1)
    ax.set(xlim=(0, 20), ylim=(0, 20), xticks=[0, 5, 10, 15, 20], yticks=[0, 5, 10, 15, 20], aspect=1,
           xlabel="WINDFOR, GW")
    style.header(ax, "Mean outturn, GW")
    ax.legend([style.line(_p["outturn"]), style.line(_p["faint"])], ["Metered + curtailed", "Metered"], loc="upper left")
    figure_target
    return


@app.cell
def _(LinearRegression, dataset, sources, table):
    _fit = LinearRegression().fit(table[["y_metered", "curtailed"]], table.windfor)
    _price = dataset.hourly(sources.prices("2024-03-09", "2024-12-31")).reindex(table.index)
    _negative = _price < 0
    _gap = table.windfor - table.y
    print(f"WINDFOR ≈ {_fit.coef_[0]:.2f}·metered + {_fit.coef_[1]:.2f}·curtailed + {_fit.intercept_:.0f} MW")
    print(f"curtailment: {(table.curtailed > 50).mean():.0%} of hours, {table.curtailed.sum() / table.y.sum():.0%} of available energy")
    print(f"negative-price hours: {_negative.mean():.1%}; WINDFOR − y there {_gap[_negative].mean():+.0f} MW, elsewhere {_gap[~_negative].mean():+.0f} MW")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assumptions that change the data

    | | assumption | effect |
    |---|---|---|
    | 1 | Target adds curtailment back | 13% of available energy; curtailment in 59% of hours |
    | 2 | Only transmission-metered wind (~25 of ~32 GW) | same scope as WINDFOR, so the comparison is fair; not total GB wind |
    | 3 | Self-curtailment at negative prices is not added back: it is not a BM action | in the 3% of negative-price hours WINDFOR sits ~570 MW above `y` (~200 MW elsewhere) |
    | 4 | Hourly value at t = mean of the two half-hours ending at t | other alignments change MAE by < 3% |
    | 5 | Weather = 10 m wind in the nearest 0.25° cell to each point, raised to hub height by a one-seventh power law; 6-hourly steps interpolated | smooths short ramps; real shear varies with stability and site |
    | 6 | Point locations come from the 2026 farm list | includes a few farms built after 2024; the model learns point weights |
    | 7 | Capacity = current nameplate from each unit's first metered output + 7 d | tracks new farms (23.3 → 25.5 GW in 2024); ignores outages |

    ## The benchmark to beat

    Scored on Aug–Dec 2024: **WINDFOR**; a **power curve**, a generic turbine curve per point
    combined by non-negative weights fitted on Apr–Jul; and their 50/50 **blend**. Weather alone
    must beat WINDFOR; weather plus WINDFOR must beat the blend. Errors persist for hours, so
    every interval in this work resamples whole days.
    """)
    return


@app.cell
def _(pd, scored):
    _errors = scored[["windfor", "powercurve", "blend"]].sub(scored.y, axis=0)
    pd.DataFrame({
        "MAE, MW": _errors.abs().mean(),
        "error autocorrelation, 1 h": [_errors[c].autocorr(1) for c in _errors],
        "error autocorrelation, 24 h": [_errors[c].autocorr(24) for c in _errors],
    }).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stakes: what WINDFOR's error costs

    The README's opening numbers, for calendar 2025. This prices NESO's own error; no model is
    scored. Each hour of WINDFOR's error against `y` is priced at the hour-ending mean of
    |imbalance price − market index price|, what a MWh costs to settle in real time rather than
    trade ahead.
    """)
    return


@app.cell
def _(dataset, evaluation, pd, sources):
    _mix = sources.generation_mix("2025-01-01", "2025-12-31").loc["2025"]
    _twh = _mix.drop(columns=[fuel for fuel in _mix if fuel.startswith("INT")]).sum() / 2 / 1e6
    _costs = sources.balancing_costs().loc["2025"].sum() / 1e6

    _spread = dataset.hourly((sources.system_prices("2025-01-01", "2025-12-31")
                              - sources.prices("2024-12-31", "2026-01-01")).abs())
    _held_out = evaluation.load_forecasts().loc["2025"]
    _error = (_held_out.windfor - _held_out.y).abs()
    _pounds = (_error * _spread.reindex(_error.index)).sum()

    pd.Series({
        "wind, TWh metered": _twh.WIND,
        "wind share of metered generation": _twh.WIND / _twh.sum(),
        "gas (CCGT + OCGT), TWh": _twh.CCGT + _twh.OCGT,
        "NESO balancing cost, £m": _costs.sum(),
        "  of which constraints, £m": _costs.Constraints,
        "  of which reserve, £m": _costs["Positive Reserve"] + _costs["Negative Reserve"],
        "WINDFOR absolute error, TWh": _error.sum() / 1e6,
        "mean |imbalance − market| spread, £/MWh": _spread.loc["2025"].mean(),
        "WINDFOR error at that spread, £m": _pounds / 1e6,
        "a 10% cut in it, £m / year": _pounds / 1e7,
    }).round(3)
    return


if __name__ == "__main__":
    app.run()

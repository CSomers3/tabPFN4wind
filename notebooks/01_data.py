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
    wind as well as NESO's own forecast, WINDFOR, issued at the same moment?

    Window: 2024-04-02 → 2024-12-31 (274 days). Anything fitted is fitted on Apr–Jul and
    scored on Aug–Dec. The 2025+ test window is not scored here. Needs the raw pulls
    (`windpfn-fetch`).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from windpfn import dataset, evaluation, forecasting, sources, style, weather

    style.use()
    WIDTH = 7.5
    _p = style.PALETTE
    NESO, CURVE, METERED, OUTTURN = _p["incumbent"], _p["ink2"], _p["muted"], _p["outturn"]

    def to_gw(ax, axis="y"):
        getattr(ax, f"{axis}axis").set_major_formatter(lambda value, _: f"{value / 1e3:g}")

    return (
        CURVE,
        LinearRegression,
        METERED,
        NESO,
        OUTTURN,
        WIDTH,
        dataset,
        evaluation,
        forecasting,
        np,
        pd,
        plt,
        sources,
        style,
        to_gw,
        weather,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audit · WINDFOR vintages

    Four per day until 2019-09, eight since. Issue times move by +1 h in UTC during BST, so
    there is no 08:30Z vintage in winter. Each vintage is hourly and repeats elapsed hours
    unchanged, so rows with `startTime <= publishTime` are dropped. A vintage is visible
    about two minutes after its publishTime (`windpfn-fetch --poll`).

    DGWS (day-ahead wind and solar) is issued at 16:45Z D-1, after our cutoff, and has a
    different scope, so it is not a usable benchmark.
    """)
    return


@app.cell
def _(sources):
    vintages, outturn = sources.windfor(), sources.fuelhh()
    per_vintage = vintages.groupby("publishTime").size()
    return outturn, per_vintage, vintages


@app.cell
def _(WIDTH, per_vintage, plt):
    _per_day = per_vintage.groupby(per_vintage.index.floor("D")).size().resample("MS").mean()
    _fig, _ax = plt.subplots(figsize=(WIDTH, 1.8))
    _ax.plot(_per_day)
    _ax.set(ylabel="vintages per day")
    _fig
    return


@app.cell
def _(pd, per_vintage):
    _issued = per_vintage["2025"].index
    pd.crosstab(_issued.strftime("%H:%M").rename("issue (UTC)"), _issued.month.rename("month, 2025"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Share of delivery days fully covered by the latest vintage at or before D-1 08:30Z:
    """)
    return


@app.cell
def _(sources, vintages):
    day_ahead = sources.day_ahead_vintage(vintages)
    _hours = day_ahead.groupby(day_ahead.index.floor("D")).size()
    (_hours == 24).groupby(_hours.index.year).mean().round(3).to_frame("full day")
    return (day_ahead,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audit · outturn

    FUELHH is keyed on UTC `startTime`; the API's settlement-date labels are wrong until
    2022-07. Clock-change days come out at 46 or 50 periods. Against 5-minute FUELINST
    (2021), MAE differs by under 3% across hour-start, centred and hour-ending alignments.
    """)
    return


@app.cell
def _(outturn):
    outturn.groupby(outturn.index.tz_convert("Europe/London").date).size().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    WINDFOR against metered outturn, 2019–21: bias is near zero in the middle deciles and
    large at high forecasts, consistent with curtailment.
    """)
    return


@app.cell
def _(NESO, WIDTH, dataset, day_ahead, outturn, pd, plt):
    _hourly = dataset.hourly(outturn).rename("y")
    _joined = pd.concat([day_ahead.generation.rename("f"), _hourly], axis=1).dropna().sort_index()
    _joined = _joined["2019":"2021"]
    _decile = pd.qcut(_joined.f, 10)
    _bias = (_joined.f - _joined.y).groupby(_decile, observed=True).mean()

    _fig, _ax = plt.subplots(figsize=(WIDTH / 2, 2))
    _ax.bar(range(10), _bias, color=NESO, width=0.7)
    _ax.set_xticks(range(10), [f"{b.mid / 1e3:.1f}" for b in _bias.index])
    _ax.set(xlabel="WINDFOR decile midpoint (GW)", ylabel="WINDFOR − outturn (MW)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audit · capacity and weather

    The trailing 365-day maximum of outturn plateaus from 2023 despite new offshore
    capacity, because constraints cap it, so it cannot serve as capacity. The BM unit list
    is an undated snapshot (285 wind units, 30.8 GW). `sources.capacity` counts each unit
    from its first metered B1610 output, plus 7 days' publication lag, at snapshot
    nameplate.

    | NWP source | Verdict |
    |---|---|
    | Open-Meteo Historical Forecast | stitches the early hours of each run, a near-analysis: **disqualified** |
    | Open-Meteo Previous Runs `_day1` | late hours of D come from runs after the cutoff: **leaks** |
    | Open-Meteo Single Runs | point-in-time, but no publication timestamps |
    | ECMWF IFS open data | point-in-time, but whole global fields: ~324 GB for every run to 90 h |
    | **ECMWF AIFS Single, dynamical.org** | point-in-time; every run lands 6h14m after initialisation in its update history: **used** |
    """)
    return


@app.cell
def _(OUTTURN, WIDTH, outturn, plt, to_gw):
    _fig, _ax = plt.subplots(figsize=(WIDTH, 1.8))
    _ax.plot(outturn.resample("1h").mean().rolling("365D").max(), color=OUTTURN)
    _ax.set(ylabel="trailing 365-day max (GW)")
    to_gw(_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When the forecast is made

    Every day D-1 we forecast the 24 hours of day D, 15 minutes after NESO publishes its
    morning WINDFOR vintage (08:30 UTC in summer, 07:30 in winter), so both forecasts share
    an information cutoff. The weather run must have been public before WINDFOR was, taking
    each AIFS run as public 6h30m after initialisation. In both seasons that is the 00Z run
    of D-1.
    """)
    return


@app.cell
def _(dataset, forecasting):
    table = forecasting.add_baselines(dataset.build("2024-04-02", "2024-12-31"))
    scored = table[table.index >= forecasting.BASELINE_SPLIT]
    return scored, table


@app.cell
def _(CURVE, NESO, WIDTH, np, plt, style, table):
    def clock(hours):
        return f"{hours % 24 // 1:02.0f}:{hours % 1 * 60:02.0f}"

    def timeline(table):
        """Median run, publication, vintage and issue times relative to D 00:00 UTC, by season."""
        days = table.groupby(table.index.normalize()).first()

        def relative(times):
            return (times - days.index).dt.total_seconds() / 3600

        days = days.assign(
            run=relative(days.run), pub=relative(days.published),
            vintage=relative(days.windfor_pub), issue=relative(days.issue_time),
        )
        seasons = [("summer (BST)", days[days.issue > -15.5]), ("winter (GMT)", days[days.issue < -15.5])]
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.2), sharex=True)
        for ax, (name, season) in zip(axes, seasons):
            run = season.run.mode()[0]
            published = season[season.run == run].pub.median()
            vintage, issue = season.vintage.median(), season.issue.median()

            ax.plot([run, published], [2, 2], color=CURVE, lw=2.5, solid_capstyle="butt")
            ax.plot(run, 2, "o", mfc=style.PALETTE["bg"], mec=CURVE, ms=5)
            ax.plot(published, 2, "o", color=CURVE, ms=5)
            ax.annotate(f"AIFS {run % 24:02.0f}Z run, public {clock(published)}", (run, 2),
                        xytext=(-4, 6), textcoords="offset points", color=CURVE)
            ax.plot(vintage, 1, "D", color=NESO, ms=4.5)
            ax.annotate(f"WINDFOR vintage {clock(vintage)}", (vintage, 1), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", color=NESO)
            ax.barh(0, 24, left=0, height=0.45, color=style.PALETTE["rule"])
            ax.text(12, 0, f"24 hourly valid times, lead {-issue:.0f}–{23 - issue:.0f} h",
                    ha="center", va="center")
            ax.axvline(issue, color=style.PALETTE["ink2"], ls=(0, (4, 3)), lw=0.7)
            ax.text(issue + 0.5, 1, f"issue {clock(issue)}", va="center", weight="bold")

            used = season.run.mod(24).value_counts(normalize=True)
            used = ", ".join(f"{hour:02.0f}Z {share:.0%}" for hour, share in used.items() if share > 0.005)
            ax.set_title(f"{name}, {len(season)} days. AIFS run used: {used}", loc="left")
            ax.set(yticks=[2, 1, 0], yticklabels=["weather", "benchmark", "delivery"],
                   ylim=(-0.6, 2.9), xlim=(-31, 24.5))
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

        ticks = np.arange(-30, 25, 6)
        day_names = {-24: "\nD−1", 0: "\nD", 24: "\nD+1"}
        axes[1].set_xticks(ticks, [f"{t % 24:02.0f}:00" + day_names.get(t, "") for t in ticks])
        axes[1].set_xlabel("UTC")
        return fig

    timeline(table)
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

    The 20 points are capacity-weighted clusters of GB wind farms (REPD, ≥ 10 MW): 10
    offshore, 10 onshore.
    """)
    return


@app.cell
def _(CURVE, METERED, OUTTURN, plt, sources, style, weather):
    _farms, _points = weather.clusters()
    _fig, _ax = plt.subplots(figsize=(3, 3.8))
    for _feature in sources.get_json("naturalearth", weather.NATURAL_EARTH)["features"]:
        if _feature["properties"]["ADM0_A3"] in ("GBR", "IRL"):
            for _polygon in _feature["geometry"]["coordinates"]:
                _lon, _lat = zip(*_polygon[0])
                _ax.fill(_lon, _lat, fc=style.PALETTE["panel"], ec=style.PALETTE["rule"], lw=0.6, zorder=0)
    _ax.scatter(*weather.to_lonlat(_farms.x.values, _farms.y.values), s=_farms.mw / 20,
                color=METERED, lw=0, alpha=0.6, label="wind farm (REPD)")
    _offshore = _points.index.str.startswith("off")
    for _mask, _colour, _label in ((_offshore, OUTTURN, "offshore point"), (~_offshore, CURVE, "onshore point")):
        _ax.scatter(_points.lon[_mask], _points.lat[_mask], s=_points.mw[_mask] / 20, color=_colour,
                    ec=style.PALETTE["bg"], lw=0.4, label=_label)
    _ax.set(aspect=1.7, xlim=(-8.5, 3), ylim=(49.8, 59.5), xticks=[], yticks=[])
    _ax.spines[["left", "bottom"]].set_visible(False)
    _ax.legend(loc="upper right", markerscale=0.6)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The table

    One row per delivery hour; `dataset.check` enforces that every column except `y` was known at
    issue. Inputs: 10 m wind speed and direction at the 20 points, capacity `cap`,
    `lead_h`, `hour`, `doy_sin`, `doy_cos`. Benchmark: `windfor`. Target: `y`, below.
    """)
    return


@app.cell
def _(dataset, table):
    table[dataset.SPEEDS + dataset.COVARIATES + ["windfor", "y"]].iloc[[0, 12]].T.round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The target: what WINDFOR forecasts

    On windy days the grid between Scotland and England fills up and NESO pays Scottish
    farms to switch off, so metered output falls short of what the wind could have made.
    WINDFOR is an input to that decision and should forecast output before curtailment.
    Against metered output WINDFOR looks biased high; with curtailment added back, the two
    sit on the diagonal (a). So the target is `y` = metered + curtailed.
    """)
    return


@app.cell
def _(CURVE, METERED, OUTTURN, WIDTH, plt, table, to_gw):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    _level = (table.windfor // 1000) * 1000 + 500
    _enough = _level.map(_level.value_counts()) >= 30
    for _column, _colour, _label in (("y_metered", METERED, "metered"), ("y", OUTTURN, "metered + curtailed")):
        _group = table[_enough].groupby(_level)[_column]
        _left.fill_between(_group.mean().index, _group.quantile(0.1), _group.quantile(0.9),
                           color=_colour, alpha=0.15, lw=0)
        _left.plot(_group.mean(), color=_colour, label=_label)
    _left.plot([0, 2e4], [0, 2e4], color=CURVE, lw=0.8, ls=(0, (4, 3)), label="y = x")
    _left.set(xlabel="WINDFOR (GW)", ylabel="outturn (GW)", xlim=(0, 2e4), ylim=(0, 2e4), aspect=1)
    to_gw(_left)
    to_gw(_left, "x")
    _left.legend(loc="upper left")

    _monthly = table.resample("MS")[["y_metered", "curtailed"]].mean()
    _x = range(len(_monthly))
    _right.bar(_x, _monthly.y_metered, color=METERED, width=0.7, label="metered")
    _right.bar(_x, _monthly.curtailed, bottom=_monthly.y_metered, color=OUTTURN, width=0.7, label="curtailed")
    for _i, (_total, _share) in enumerate(zip(_monthly.sum(1), _monthly.curtailed / _monthly.sum(1))):
        _right.text(_i, _total + 150, f"{_share:.0%}", ha="center", fontsize=6)
    _right.set_xticks(list(_x), _monthly.index.strftime("%b"))
    _right.set(ylabel="mean available wind (GW)")
    to_gw(_right)
    _right.legend(loc="upper left")
    for _ax, _letter in zip((_left, _right), "ab"):
        _ax.set_title(_letter, loc="left", weight="bold")
    _fig
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
    | 1 | Target adds curtailment back | +12% energy; changes 58% of hours |
    | 2 | Only transmission-metered wind (~25 of ~32 GW) | same scope for target and WINDFOR, so the comparison is fair; not total GB wind |
    | 3 | Self-curtailment at negative prices is not added back: it is not a BM action | in the 3% of negative-price hours WINDFOR sits ~600 MW above `y` (~190 MW elsewhere) |
    | 4 | Hourly value at t = mean of the two half-hours ending at t | other alignments change MAE by < 3% |
    | 5 | Weather = 10 m wind in the nearest 0.25° cell to each point, raised to hub height by a one-seventh power law; 6-hourly steps interpolated | smooths short ramps; real shear varies with stability and site |
    | 6 | Only runs public before WINDFOR, each taken as public 6h30m after initialisation | dynamical.org keeps no per-run publication time |
    | 7 | Point locations come from the 2026 farm list | includes a few farms built after 2024; the model learns point weights |
    | 8 | Capacity = today's nameplate from each unit's first metered output + 7 d | tracks new farms (23.3 → 25.5 GW in 2024); ignores outages |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The benchmark to beat

    Two references, scored on Aug–Dec: **WINDFOR**, NESO's operational forecast, and a
    **power curve**, the simplest physical model: a generic turbine curve per point,
    combined by one non-negative weight each, fitted on Apr–Jul.
    """)
    return


@app.cell
def _(CURVE, NESO, OUTTURN, WIDTH, plt, table, to_gw):
    _week = table["2024-11-01":"2024-11-14"]
    _fig, _ax = plt.subplots(figsize=(WIDTH, 2.2))
    _ax.plot(_week.y, color=OUTTURN, label="target")
    _ax.plot(_week.windfor, color=NESO, lw=0.9, label="WINDFOR (NESO)")
    _ax.plot(_week.powercurve, color=CURVE, lw=0.9, ls=(0, (4, 2)), label="power curve (AIFS)")
    _ax.set(ylabel="GB wind (GW)", ylim=(0, None), xlim=(_week.index[0], _week.index[-1]))
    to_gw(_ax)
    _ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    _fig
    return


@app.cell
def _(CURVE, METERED, NESO, WIDTH, np, pd, plt, scored):
    FORECASTS = {"windfor": ("WINDFOR", NESO), "powercurve": ("power curve", CURVE), "blend": ("50/50 blend", METERED)}

    def mae(frame, column):
        return (frame[column] - frame.y).abs().mean()

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    _bars = _left.bar(range(3), [mae(scored, c) for c in FORECASTS], width=0.7,
                      color=[colour for _, colour in FORECASTS.values()])
    _left.bar_label(_bars, fmt="%.0f", fontsize=6, padding=2)
    _left.set_xticks(range(3), [name for name, _ in FORECASTS.values()])
    _left.set(ylabel="MAE, Aug–Dec (MW)")

    _level = pd.qcut((scored.windfor + scored.powercurve) / 2, 4, labels=["low", "mid", "high", "very high"])
    _by_level = scored.groupby(_level, observed=True).apply(
        lambda g: pd.Series({c: mae(g, c) for c in FORECASTS}), include_groups=False
    )
    for _i, (_column, (_name, _colour)) in enumerate(FORECASTS.items()):
        _right.bar(np.arange(4) + (_i - 1) * 0.27, _by_level[_column], 0.27, color=_colour, label=_name)
    _right.set_xticks(range(4), _by_level.index)
    _right.set(xlabel="forecast level (quartile, known at issue)", ylabel="MAE (MW)")
    _right.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    WINDFOR beats the power curve by 5%, yet their errors are only moderately correlated,
    so a plain 50/50 average beats both by 13%, at every forecast level. That sets two
    bars: **weather only** must beat WINDFOR; **weather + WINDFOR** must beat the blend.

    ## What the errors look like

    (a) Error spread grows with output, so a fixed ± band would be wrong. (b) Error grows
    with lead time; WINDFOR's edge is at short leads. (c) Errors persist for hours, so
    significance tests resample whole days.
    """)
    return


@app.cell
def _(CURVE, METERED, NESO, WIDTH, pd, plt, scored, table, to_gw):
    _fig, _axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    _level = pd.qcut(table.powercurve, 10)
    for _column, _colour, _name in (("windfor", NESO, "WINDFOR"), ("powercurve", CURVE, "power curve")):
        _error = table[_column] - table.y
        _axes[0].plot([b.mid for b in _level.cat.categories], _error.groupby(_level, observed=True).std(),
                      "o-", color=_colour, label=_name)
        _axes[1].plot((scored[_column] - scored.y).abs().groupby(scored.lead_h).mean(), color=_colour)
        _axes[2].plot(range(73), [_error.autocorr(k) for k in range(73)], color=_colour)
    to_gw(_axes[0], "x")
    _axes[0].set(xlabel="forecast level (GW)", ylabel="error s.d. (MW)", ylim=(0, None))
    _axes[0].legend(loc="lower right")
    _axes[1].set(xlabel="lead time (h)", ylabel="MAE, Aug–Dec (MW)")
    _axes[2].axhline(0, color=METERED, lw=0.6)
    _axes[2].set(xlabel="lag (h)", ylabel="error autocorrelation", xticks=range(0, 73, 24))
    for _ax, _letter in zip(_axes, "abc"):
        _ax.set_title(_letter, loc="left", weight="bold")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stakes: what WINDFOR's error costs

    The README's opening numbers, for calendar 2025. This prices NESO's own error; no model
    is scored. Each hour of WINDFOR's error against `y` is priced at the hour-ending mean of
    |imbalance price − market index price|, what a MWh costs to settle at real time rather
    than trade ahead.
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

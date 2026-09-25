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
    # 05 · Live

    A still of the live page's chart (`explorer.py`), drawn the same way: outturn since
    yesterday, then the most recent forecasts from TabPFN-3.5 (its full distribution, p1–p99)
    and NESO, from the last settled hour to the end of tomorrow. Reads
    `notebooks/public/live.csv` and pulls outturn from Elexon.

    Produces the README figure `live.png`.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from windpfn import cli, dataset, sources, style

    return cli, dataset, np, pd, plt, sources, style


@app.cell
def _(cli, dataset, pd, sources):
    live = pd.read_csv(cli.LIVE, index_col="hour", parse_dates=["hour", "issue_time"])
    now = pd.Timestamp.now("UTC").tz_convert(None)
    start = now.normalize() - pd.Timedelta(days=1)
    _first, _last = f"{start - pd.Timedelta(days=1):%Y-%m-%d}", f"{now:%Y-%m-%d}"
    _available = sources.fuelhh(_first, _last) + sources.curtailment(_first, _last, sources.wind_units().index)
    observed = dataset.hourly(_available).tz_convert(None)
    observed = observed[observed.index.minute == 0][start:now]
    return live, now, observed, start


@app.cell
def _(live, now, np, observed, pd, plt, sources, start, style):
    _p = style.PAPER
    joined = observed.index[-1]
    ahead = live.loc[joined:].dropna(subset=["q50"])
    hours = pd.date_range(start, min(start + pd.Timedelta(hours=71), ahead.index[-1]), freq="h")
    forecast = ahead.drop(columns="issue_time").reindex(hours) / 1e3
    incumbent = forecast.windfor
    ceiling = max(10, np.ceil(np.nanmax([forecast.q99.max(), incumbent.max(), observed.max() / 1e3]) / 5) * 5)

    with style.paper():
        figure, ax = plt.subplots(figsize=(6.4, 2.6))
        ax.axvspan(start, now, color=_p["past"], lw=0, zorder=0)
        for _lower in range(1, 50):
            ax.fill_between(hours, forecast[f"q{_lower}"], forecast[f"q{100 - _lower}"], color=style.FAN(_lower / 50), lw=0)
        ax.plot(forecast.q50, color=_p["bg"])
        ax.plot(incumbent, color=_p["incumbent"])
        ax.plot(observed / 1e3, color=_p["outturn"], lw=1.4)
        ax.plot(joined, observed.iloc[-1] / 1e3, "o", ms=4, color=_p["outturn"], mec=_p["bg"], mew=1.2)
        ax.axhline(0, color=_p["rule"], lw=0.6)
        ax.vlines(now, 0, ceiling * 1.03, color=_p["faint"], lw=0.6, ls=(0, (4, 3)), clip_on=False)
        ax.text(now, ceiling * 1.05, "Now", ha="center", va="bottom", color=_p["muted"])

        _last = forecast.q50.last_valid_index()
        _labelled = [99, 90, 50, 10, 1]
        _edges = [forecast.at[_last, f"q{_q}"] for _q in _labelled]
        _gap = ceiling * 0.055
        _ys = list(_edges)
        for _k in range(1, -1, -1):
            _ys[_k] = max(_ys[_k], _ys[_k + 1] + _gap)
        for _k in range(3, 5):
            _ys[_k] = min(_ys[_k], _ys[_k - 1] - _gap)
        for _q, _edge, _y in zip(_labelled, _edges, _ys):
            ax.annotate(f"p{_q}", (_last, _edge), xytext=(_last + pd.Timedelta(hours=2.2), _y), va="center",
                        fontsize=6.5, color=_p["ink"] if _q == 50 else _p["muted"],
                        fontweight="semibold" if _q == 50 else "normal", annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": _p["rule"], "lw": 0.6, "shrinkA": 0, "shrinkB": 1})

        ax.set(xlim=(hours[0], hours[-1]), ylim=(0, ceiling), yticks=np.arange(0, ceiling + 1, 5))
        for _day in hours[hours.hour == 0][1:]:
            ax.axvline(_day, color=_p["rule2"], lw=0.6, zorder=0.5)
        _ticks = hours[hours.hour % 6 == 0]
        ax.set_xticks(_ticks, [f"{_t:%a %d %b}" if _t.hour == 0 else f"{_t:%H:%M}" for _t in _ticks])
        for _t, _label in zip(_ticks, ax.get_xticklabels()):
            if _t.hour == 0:
                _label.set(color=_p["ink"], fontweight="semibold")

        style.header(ax, "Wind generation, GW · UTC",
                     [style.line(_p["outturn"], 1.4), style.Fan(), style.line(_p["incumbent"])],
                     ["Outturn", "TabPFN-3.5", "NESO"], y=1.12)
        figure.savefig(sources.ROOT / "notebooks" / "live.png")
    figure
    return


if __name__ == "__main__":
    app.run()

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
    yesterday under the forecasts from TabPFN-3.5 (its p10–p90 to p40–p60 fan and median)
    and NESO, each hour from the last update issued before it, to the end of tomorrow. Reads
    `notebooks/public/live.csv` and pulls outturn from Elexon.

    Produces the README figure `live.png`.
    """)
    return


@app.cell
def _():
    import pandas as pd

    from windpfn import cli, dataset, sources, style

    return cli, dataset, pd, sources, style


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
def _(live, now, observed, pd, sources, start, style):
    _issued = live.dropna(subset=["q50"])
    _hours = pd.date_range(start, min(start + pd.Timedelta(hours=71), _issued.index[-1]), freq="h")
    _forecast = _issued.drop(columns="issue_time").reindex(_hours) / 1e3
    style.fan_chart(_forecast, _forecast.windfor, observed / 1e3, now, path=sources.ROOT / "notebooks" / "live.png")
    return


if __name__ == "__main__":
    app.run()

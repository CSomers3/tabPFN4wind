import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import pandas as pd

    from windpfn import dataset, evaluation, forecasting, style

    return dataset, evaluation, forecasting, mo, os, pd, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # windpfn on the bundled sample

    `data/sample/` holds every input from 2026-03-01 to 2026-09-15: NESO's WINDFOR, metered wind
    plus curtailment, capacity, and ECMWF AIFS weather at 20 points. This notebook forecasts each
    day from 2026-08-01 a day ahead, as the held-out test did, and scores it against WINDFOR.

    TabPFN-3.5 runs on Prior Labs' hosted API and needs `TABPFN_TOKEN` set before marimo starts.
    Without it, only the two references run. The context here is months, not years, so the scores
    are a smoke test, not a result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the model sees

    One row per hour: capacity factor from a power curve and wind direction at each point,
    capacity, lead time, calendar, and NESO's forecast.
    """)
    return


@app.cell
def _(dataset, forecasting):
    table = dataset.build(dataset.SAMPLE_START, dataset.SAMPLE_END, dataset.load())
    inputs = forecasting.features(table).join(table.windfor)
    inputs
    return inputs, table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecast

    Each month is forecast from every hour settled a day before its first issue.
    """)
    return


@app.cell
def _(dataset, forecasting, inputs, os, table):
    forecasters = {"conformal": forecasting.conformal, "lgbm+windfor": forecasting.lightgbm}
    if os.environ.get("TABPFN_TOKEN"):
        forecasters["tabpfn+windfor"] = forecasting.tabpfn
    runs = {
        name: forecasting.backtest(inputs, table, dataset.SAMPLE_TEST_START, forecaster=forecaster)
        for name, forecaster in forecasters.items()
    }
    forecasts = table.loc[dataset.SAMPLE_TEST_START :, ["y", "windfor"]].join(forecasting.pack(runs))
    return forecasts, runs


@app.cell
def _(evaluation, forecasts, runs):
    LABELS = {
        "windfor": "NESO",
        "conformal": "NESO + conformal band",
        "lgbm+windfor": "LightGBM",
        "tabpfn+windfor": "TabPFN-3.5",
    }
    _, _, _scores = evaluation.scorecard(forecasts, list(runs))
    _scores.rename(index=LABELS)[["MAE", "ΔMAE", "lo", "hi", "CRPS", "cover80", "winkler80"]].round(3)
    return (LABELS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MW. ΔMAE is against NESO, with its 95% day-block bootstrap interval in `lo` and `hi`.
    `cover80` is the share of hours inside the 10–90% interval.
    """)
    return


@app.cell
def _(forecasts, mo, runs):
    shown = mo.ui.dropdown(list(runs), value=list(runs)[-1], label="Forecast")
    week = mo.ui.dropdown(
        {f"{day:%d %b}": day for day in forecasts.index.floor("D").unique()[::7]},
        value=f"{forecasts.index[0]:%d %b}",
        label="Week from",
    )
    mo.hstack([shown, week], justify="start")
    return shown, week


@app.cell
def _(LABELS, forecasting, forecasts, pd, shown, style, week):
    _hours = forecasts.loc[week.value : week.value + pd.Timedelta(days=7, hours=-1)]
    _deciles = forecasting.unpack(_hours)[shown.value]
    style.fan_chart(_deciles / 1e3, _hours.windfor / 1e3, _hours.y / 1e3, label=LABELS[shown.value])
    return


if __name__ == "__main__":
    app.run()

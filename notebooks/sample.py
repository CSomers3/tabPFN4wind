import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import pandas as pd

    from windpfn import dataset, evaluation, forecasting, style

    style.use()
    return dataset, evaluation, forecasting, mo, os, pd, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # windpfn on the bundled sample

    The whole pipeline on `data/sample/`, which holds every input from 2026-03-01 to 2026-09-15.
    Each day from 2026-08-01 is forecast a day ahead, as in the held-out test, and scored against
    NESO. TabPFN-3.5 runs on Prior Labs' hosted API when `TABPFN_TOKEN` is set; without it, only
    the two references run. The context is months, not years, so the scores are a smoke test.

    Each row is one hour: capacity factor from a power curve and wind direction at 20 points,
    capacity, lead time, calendar, and NESO's forecast.
    """)
    return


@app.cell
def _(dataset, forecasting):
    table = dataset.build(dataset.SAMPLE_START, dataset.SAMPLE_END, dataset.load())
    inputs = forecasting.features(table).join(table.windfor)
    inputs
    return inputs, table


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
    LABELS = {"windfor": "NESO", "conformal": "NESO + conformal band", "lgbm+windfor": "LightGBM", "tabpfn+windfor": "TabPFN-3.5"}
    _, _, _scores = evaluation.scorecard(forecasts, list(runs))
    _scores.loc[["windfor", *runs],["MAE", "ΔMAE", "lo", "hi", "CRPS", "cover80", "winkler80"]].rename(index=LABELS).round(3)
    return (LABELS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MW. ΔMAE is against NESO, with its 95% day-block bootstrap interval in `lo` and `hi`; `cover80`
    is the share of hours inside p10–p90.
    """)
    return


@app.cell
def _(forecasts, mo):
    week = mo.ui.dropdown(
        {f"{day:%d %b}": day for day in forecasts.index.floor("D").unique()[::7]},
        value=f"{forecasts.index[0]:%d %b}",
        label="Week from",
    )
    week
    return (week,)


@app.cell
def _(LABELS, forecasting, forecasts, pd, runs, style, week):
    _shown = list(runs)[-1]
    _hours = forecasts.loc[week.value : week.value + pd.Timedelta(days=7, hours=-1)]
    _deciles = forecasting.unpack(_hours)[_shown]
    style.fan_chart(_deciles / 1e3, _hours.windfor / 1e3, _hours.y / 1e3, label=LABELS[_shown])
    return


if __name__ == "__main__":
    app.run()

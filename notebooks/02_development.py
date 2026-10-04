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
    # 02 · Development: TabPFN-3.5 vs NESO, 2024

    Features, target scaling, context and embargo were chosen here, on 2024 only, before
    the protocol was frozen. TabPFN-3.5 reads every hour settled a day before issue as its
    context, refreshed monthly, and predicts capacity factor from the point power curves,
    wind direction and calendar. `tabpfn+windfor` also sees NESO's forecast.

    Scored Jul–Dec 2024, so the first month has three months of context; ΔMAE is vs the
    first row with a 95% day-block bootstrap interval. Reads the committed history in
    `data/history/` and calls the hosted TabPFN API, so it needs a tabpfn-client token
    (about 1M tokens for everything below).
    """)
    return


@app.cell
def _():
    from functools import partial

    import pandas as pd

    from windpfn import dataset, evaluation, forecasting, sources, weather

    START, END = "2024-07-01", "2024-12-31"
    return END, START, dataset, evaluation, forecasting, partial, pd, sources, weather


@app.cell
def _(END, START, dataset, evaluation, forecasting, sources):
    inputs = dataset.load(dataset.HISTORY)
    table = dataset.build(sources.ARCHIVE_START, END, inputs)
    features = forecasting.features(table)
    development = {
        name: forecasting.backtest(x, table, START, END)
        for name, x in {"tabpfn": features, "tabpfn+windfor": features.join(table.windfor)}.items()
    }
    _errors = table.loc[START:END, ["windfor"]].assign(**{n: r.q50 for n, r in development.items()}).sub(table.y[START:END], axis=0)
    evaluation.point_scores(_errors)
    return development, features, inputs, table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior Labs' performance guidance, tested

    Each row changes one thing in `tabpfn+windfor`, following
    [Improving performance](https://docs.priorlabs.ai/improving-performance):

    - `windfor as capacity factor`: WINDFOR divided by capacity, on the target's scale (a ratio feature).
    - `fleet power curve`: the capacity-weighted mean of the 20 point curves (an aggregate).
    - `native datetime`: valid time as a datetime column in place of hour and day-of-year sin/cos.
    - `thinking`: TabPFN-3.5-Thinking, MAE objective, with and without valid time as `time_col`.
      Thinking regression returns only the mean, so it is compared with the standard mean.

    `n_estimators` above 8 is refused by the hosted API.
    """)
    return


@app.cell
def _(END, START, development, evaluation, features, forecasting, partial, pd, table, weather):
    def _mean(features, table, train, test, **settings):
        """TabPFN-3.5's predictive mean, the only output Thinking gives for regression."""
        from tabpfn_client import TabPFNRegressor

        model = TabPFNRegressor.create_default_for_version("v3.5", **settings)
        model.fit(features[train], (table.y / table.cap)[train])
        return pd.DataFrame({"q50": model.predict(features[test]) * table.cap[test].to_numpy()}, table.index[test])

    _thinking = partial(_mean, thinking_mode=True, thinking_metric="mae")

    _base = features.join(table.windfor)
    _curves = forecasting.power_curve(table).to_numpy()
    _fleet = _curves @ weather.POINTS.mw.to_numpy() / weather.POINTS.mw.sum()
    _timed = _base.assign(valid_time=table.index.tz_convert(None))
    _variants = {
        "windfor as capacity factor": (_base.drop(columns="windfor").assign(windfor_cf=table.windfor / table.cap), forecasting.tabpfn),
        "fleet power curve": (_base.assign(fleet_pc=_fleet), forecasting.tabpfn),
        "native datetime": (_timed.drop(columns=["hour", "doy_sin", "doy_cos"]), forecasting.tabpfn),
        "standard, mean": (_base, _mean),
        "thinking, mean": (_base, _thinking),
        "thinking + time_col, mean": (_timed, partial(_thinking, time_col="valid_time")),
    }
    _runs = {"tabpfn+windfor": development["tabpfn+windfor"]}
    _runs |= {name: forecasting.backtest(x, table, START, END, forecaster=f) for name, (x, f) in _variants.items()}
    _y = table.y[START:END]
    _errors = pd.DataFrame({name: run.q50 for name, run in _runs.items()}).sub(_y, axis=0)
    _probabilistic = {name: run for name, run in _runs.items() if "q10" in run}
    _scores = evaluation.point_scores(_errors, reference="tabpfn+windfor")
    _scores.join(evaluation.probabilistic_scores(_probabilistic, _y).astype(float)).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Context for the live forecast

    The live forecast reads every WINDFOR vintage, about 400,000 rows by 2026, and is cut to
    50,000 for cost. Here on Nov–Dec 2024 (about 100,000 rows of context): one uniform draw
    of 50,000, a fresh 50,000 for each of the 8 estimators (`SUBSAMPLE_SAMPLES`), and the
    full context.
    """)
    return


@app.cell
def _(END, dataset, evaluation, forecasting, inputs, partial, pd, sources):
    _first = "2024-11-01"
    _vintages = dataset.vintages(sources.ARCHIVE_START, END, inputs)
    _x = forecasting.features(_vintages).assign(
        windfor=_vintages.windfor.to_numpy(), metered_now=_vintages.metered_now.to_numpy()
    )
    _forecasters = {
        "one uniform draw": forecasting.subsampled(forecasting.tabpfn),
        "per estimator": partial(forecasting.tabpfn, inference_config={"SUBSAMPLE_SAMPLES": forecasting.CONTEXT_ROWS}),
        "full context": forecasting.tabpfn,
    }
    _rows = _vintages.loc[_first:END].reset_index(drop=True)
    _runs = {
        name: forecasting.backtest(_x, _vintages, _first, END, forecaster=f).reset_index(drop=True)
        for name, f in _forecasters.items()
    }
    _days = _vintages.loc[_first:END].index.tz_convert(None)
    _errors = pd.DataFrame({name: run.q50 for name, run in _runs.items()}).sub(_rows.y, axis=0).set_axis(_days)
    evaluation.point_scores(_errors, reference="one uniform draw").join(
        evaluation.probabilistic_scores(_runs, _rows.y).astype(float).set_axis(list(_runs))
    ).round(3)
    return


if __name__ == "__main__":
    app.run()

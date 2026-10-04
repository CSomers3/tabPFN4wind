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
    # 04 · Reference forecasts

    Added after the one-shot test in `03`, to ask whether the gain comes from TabPFN or
    from the features and protocol. Same features, target, context and embargo, and the same
    AIFS weather as TabPFN's `aifs` run; the TabPFN forecasts are read from `data/`.

    - `lgbm`: quantile LightGBM, one model per decile, hyperparameters chosen on 2024 only.
    - `conformal`: WINDFOR plus its empirical error deciles within WINDFOR-level quintiles.
      Its median is WINDFOR recalibrated by level, without weather.

    The held-out reference forecasts were produced by `windpfn-backtest references` and
    committed in `data/`. The tuning below runs locally and needs the raw pulls.
    """)
    return


@app.cell
def _():
    from functools import partial

    from windpfn import dataset, evaluation, forecasting

    return dataset, evaluation, forecasting, partial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LightGBM settings, chosen by CRPS on Apr–Dec 2024 (`forecasting.LGBM_PARAMS` holds the winner)
    """)
    return


@app.cell
def _(dataset, evaluation, forecasting, partial):
    table = dataset.build("2024-04-02", "2024-12-31")
    _inputs = forecasting.features(table).join(table.windfor)
    _grid = [
        {"num_leaves": leaves, "n_estimators": trees, "learning_rate": rate}
        for leaves in (7, 15, 31)
        for trees, rate in ((300, 0.05), (1000, 0.02))
    ]
    _window = ("2024-04-01", "2024-12-31")
    _runs = {
        str(settings): forecasting.backtest(
            _inputs, table, *_window, forecaster=partial(forecasting.lightgbm, **settings)
        )
        for settings in _grid
    }
    evaluation.probabilistic_scores(_runs, table.y[_window[0] : _window[1]]).sort_values("CRPS").round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Held-out scorecard

    CRPS is twice the mean pinball loss over the nine deciles; for a point forecast it
    equals MAE. The Winkler score is the 10–90% width plus 2/α times the distance by which
    outturn falls outside it (α = 0.2): one number for "narrow *and* covering".
    """)
    return


@app.cell
def _(evaluation):
    _, errors, scores = evaluation.scorecard(evaluation.load_forecasts())
    scores.round(3)
    return errors, scores


@app.cell
def _(scores):
    for _tabpfn in ("tabpfn+windfor ifs", "tabpfn+windfor aifs"):
        _gain = -scores.loc[_tabpfn, "ΔMAE"]
        for _name in ("conformal", "lgbm+windfor"):
            print(f"{_name} recovers {-scores.loc[_name, 'ΔMAE']:.0f} of {_tabpfn}'s {_gain:.0f} MW")
        _narrower = 1 - scores.loc[_tabpfn, "width80"] / scores.loc["conformal", "width80"]
        print(f"{_tabpfn}'s 10–90% interval is {_narrower:.0%} narrower than conformal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LightGBM against TabPFN directly on the same AIFS inputs, positive where LightGBM is worse:
    """)
    return


@app.cell
def _(errors, evaluation):
    evaluation.point_scores(errors[["tabpfn+windfor aifs", "lgbm+windfor"]], reference="tabpfn+windfor aifs")
    return


if __name__ == "__main__":
    app.run()

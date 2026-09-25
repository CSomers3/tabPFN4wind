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

    Scored Apr–Dec 2024; ΔMAE is vs WINDFOR with a 95% day-block bootstrap interval. Calls
    the hosted TabPFN API, so it needs a tabpfn-client token.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    from windpfn import dataset, evaluation, forecasting, style

    style.use()
    WIDTH = 7.5
    return WIDTH, dataset, evaluation, forecasting, pd, plt, style


@app.cell
def _(dataset, forecasting):
    table = dataset.build("2024-04-02", "2024-12-31")
    _features = forecasting.features(table)
    inputs = {"tabpfn": _features, "tabpfn+windfor": _features.join(table.windfor)}
    return inputs, table


@app.cell
def _(evaluation, forecasting, inputs, table):
    START = "2024-04-01"
    development = table[START:].assign(
        **{name: forecasting.backtest(features, table, START).q50 for name, features in inputs.items()}
    )
    errors = development[["windfor", *inputs]].sub(development.y, axis=0)
    evaluation.point_scores(errors)
    return development, errors


@app.cell
def _(WIDTH, errors, plt, style):
    _monthly = errors.abs().resample("MS").mean()
    _fig, _ax = plt.subplots(figsize=(WIDTH, 2.4))
    for _run, _colour in zip(_monthly, [style.PALETTE[c] for c in ("incumbent", "ink2", "forecast")]):
        _ax.plot(_monthly.index, _monthly[_run], "o-", ms=2.5, color=_colour, label=_run)
    _ax.set(ylabel="MAE (MW)", ylim=(0, None))
    _ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    _fig
    return


@app.cell
def _(development, errors, pd):
    errors.abs().groupby(pd.cut(development.lead_h, [15, 21, 27, 33, 40]), observed=True).mean().round(0)
    return


if __name__ == "__main__":
    app.run()

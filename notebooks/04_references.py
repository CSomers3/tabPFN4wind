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
    # 04 · Reference forecasters

    Added after the one-shot test in `03`, to ask whether the gain comes from TabPFN-3.5 or
    from the features and protocol.

    - `conformal`: WINDFOR plus its empirical error deciles within WINDFOR-level quintiles.
      Its median is WINDFOR recalibrated by level, without weather.
    - The ablation: LightGBM and a quantile regression forest, each forecasting the nine
      deciles of capacity factor from TabPFN's exact inputs (the AIFS run, rebuilt from `data/history/`), context and monthly refit.
      Hyperparameters were chosen by CRPS on Jul–Dec 2024 only.
    - Context size: TabPFN-3.5 and the two references with each month's context subsampled
      to 250–4,000 rows.

    The held-out forecasts were produced by `windpfn-backtest references`, `ablation` and
    `context`, and committed in `data/`. The tuning below runs locally from `data/history/`.
    """)
    return


@app.cell
def _():
    from functools import partial

    import matplotlib.pyplot as plt
    import pandas as pd

    from windpfn import cli, dataset, evaluation, forecasting, sources, style

    style.use()
    return cli, dataset, evaluation, forecasting, partial, pd, plt, sources, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings, chosen by CRPS on Jul–Dec 2024 (the winners are in `forecasting`)
    """)
    return


@app.cell
def _(dataset, evaluation, forecasting, partial, pd):
    GRIDS = {
        "lightgbm": [{"num_leaves": l, "n_estimators": n, "learning_rate": r} for l in (7, 15, 31) for n, r in ((300, 0.05), (1000, 0.02))],
        "qrf": [{"min_samples_leaf": m, "max_features": f} for m in (1, 10, 50) for f in (0.33, 1.0)],
    }
    _table = dataset.build("2024-04-02", "2024-12-31", dataset.load(dataset.HISTORY))
    _inputs = forecasting.features(_table).join(_table.windfor)
    _window = ("2024-07-01", "2024-12-31")
    _y = _table.y[_window[0] : _window[1]]
    _tuning = {}
    for _name, (_, _forecaster) in forecasting.REFERENCES.items():
        _runs = {
            str(_settings): forecasting.backtest(_inputs, _table, *_window, forecaster=partial(_forecaster, **_settings))
            for _settings in GRIDS[_name]
        }
        _tuning[_name] = evaluation.probabilistic_scores(_runs, _y).astype(float).sort_values("CRPS")
    pd.concat(_tuning).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Held-out ablation: identical inputs, 14,952 hours
    """)
    return


@app.cell
def _(cli, evaluation):
    forecasts = evaluation.load_forecasts()
    _, errors, scores = evaluation.scorecard(forecasts, list(cli.ABLATION))
    scores.loc[list(cli.ABLATION)].rename(index=cli.ABLATION).round(3)
    return errors, forecasts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each reference against TabPFN-3.5 directly, ΔMAE with a 95% day-block bootstrap interval,
    positive where the reference is worse:
    """)
    return


@app.cell
def _(cli, errors, evaluation):
    evaluation.point_scores(errors[list(cli.ABLATION)], reference="tabpfn+windfor aifs").rename(index=cli.ABLATION)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Context size

    MAE of each median as the monthly context is cut to a uniform sample of 250–4,000 settled
    hours (one draw per month, seed 0). `all` is the full context: 6,944 hours in January 2025,
    21,537 by September 2026.
    """)
    return


@app.cell
def _(cli, evaluation, forecasts):
    scaling = evaluation.context_scaling(forecasts)
    scaling.rename(columns=cli.ABLATION).round(0)
    return (scaling,)


@app.cell
def _(cli, forecasts, plt, scaling, sources, style):
    _p = style.PALETTE
    _x = range(len(scaling))
    _neso = (forecasts.windfor - forecasts.y).abs().mean()
    figure_scaling, _ax = plt.subplots(figsize=(style.WIDTH, 2.4))
    for _name in scaling.columns[1:]:
        _ax.plot(_x, scaling[_name], color=_p["faint"], lw=0.9, marker="o", ms=2.5)
        _above = scaling[_name].iloc[-1] == scaling.iloc[-1, 1:].max()
        _ax.annotate(cli.ABLATION[_name], (_x[-1], scaling[_name].iloc[-1]), xytext=(5, 4 if _above else -4),
                     textcoords="offset points", va="center", color=_p["muted"], fontsize=6)
    _ax.plot(_x, scaling.iloc[:, 0], color=_p["median"], lw=1.6, marker="o", ms=3.5, mec=_p["bg"], mew=1)
    _ax.axhline(_neso, color=_p["incumbent"], lw=0.9, ls=(0, (4, 3)))
    _ax.set_xticks(list(_x), [f"{size:,}" if size != "all" else "all" for size in scaling.index])
    _ax.set_xlabel("Context rows per monthly fit")
    style.header(_ax, "Held-out MAE, MW", [style.line(_p["median"], 1.6), style.line(_p["faint"]),
                                           style.line(_p["incumbent"])], ["TabPFN-3.5", "References", "NESO"])
    figure_scaling.savefig(sources.ROOT / "assets" / "scaling.png")
    figure_scaling
    return


if __name__ == "__main__":
    app.run()

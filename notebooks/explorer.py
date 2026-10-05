# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "pandas"]
#
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="GB wind · TabPFN-3.5")


@app.cell(hide_code=True)
def _():
    import asyncio
    import io
    import json
    import sys
    import urllib.request
    from urllib.parse import urlencode

    import marimo as mo
    import numpy as np
    import pandas as pd

    return asyncio, io, json, mo, np, pd, sys, urlencode, urllib


@app.cell(hide_code=True)
async def _(io, json, mo, pd, sys):
    # Written by `windpfn-page` and `windpfn-live`: windpfn itself can't be installed under Pyodide.
    # Read as text first: pandas would gunzip what the browser has already decompressed.
    public = mo.notebook_location() / "public"

    async def read(name: str) -> str:
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            return await (await pyfetch(str(public / name))).string()
        return (public / name).read_text(encoding="utf-8")

    meta = pd.Series(json.loads(await read("meta.json")))
    live = pd.read_csv(io.StringIO(await read("live.csv")), index_col="hour", parse_dates=["hour", "issue_time"])
    return live, meta


@app.cell(hide_code=True)
def _(asyncio, json, meta, pd, sys, urlencode, urllib):
    ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"

    def utc(stamps) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_convert(None)

    async def get_json(url: str):
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            return await (await pyfetch(url)).json()
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)

    async def outturn(first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
        """Hourly available wind in MW, as windpfn builds its target: metered transmission output
        plus balancing-mechanism curtailment of the wind units, hour-ending."""
        dates = pd.date_range(first - pd.Timedelta(days=1), last.normalize())
        units = urlencode([("bmUnit", unit) for unit in meta.units])
        urls = [
            f"{ELEXON}/datasets/FUELHH/stream?fuelType=WIND"
            f"&settlementDateFrom={dates[0]:%Y-%m-%d}&settlementDateTo={dates[-1]:%Y-%m-%d}"
        ] + [
            f"{ELEXON}/balancing/settlement/acceptance/volumes/all/{side}/{date:%Y-%m-%d}?{units}"
            for date in dates
            for side in ("bid", "offer")
        ]
        metered, *volumes = await asyncio.gather(*map(get_json, urls))

        metered = pd.DataFrame(metered).sort_values("publishTime").drop_duplicates("startTime", keep="last")
        metered = pd.Series(metered.generation.to_numpy(float), utc(metered.startTime)).sort_index()
        volumes = pd.DataFrame([row for response in volumes for row in response["data"]])
        removed = pd.Series(0.0, metered.index)
        if len(volumes):
            removed = removed.add(-2 * volumes.groupby(utc(volumes.startTime)).totalVolumeAccepted.sum(), fill_value=0)
        available = (metered + removed).dropna()
        hourly = (available.shift(1, freq="30min") + available.shift(2, freq="30min")) / 2
        return hourly[hourly.index.minute == 0].dropna()[first:last]

    async def latest_update(now: pd.Timestamp) -> pd.Timestamp:
        """When NESO last published WINDFOR."""
        url = (f"{ELEXON}/datasets/WINDFOR/stream?publishDateTimeFrom={now - pd.Timedelta(hours=12):%Y-%m-%dT%H:%MZ}"
               f"&publishDateTimeTo={now + pd.Timedelta(hours=1):%Y-%m-%dT%H:%MZ}")
        return utc([row["publishTime"] for row in await get_json(url)]).max()

    return latest_update, outturn


@app.cell(hide_code=True)
async def _(latest_update, live, outturn, pd):
    # Each hour pairs TabPFN-3.5 with the NESO update it read, published 15 minutes before it issued.
    incumbent = pd.DataFrame({"generation": live.windfor, "published": live.issue_time - pd.Timedelta("15min")})
    now = pd.Timestamp.now("UTC").tz_convert(None)
    days = pd.date_range(now.normalize() - pd.Timedelta(days=1), periods=3)
    try:
        observed = await outturn(days[0], now)
    except Exception:
        observed = None
    try:
        newest = await latest_update(now)
    except Exception:
        newest = None
    return days, incumbent, newest, now, observed


@app.cell(hide_code=True)
def _():
    P = {
        "bg": "#FFFFFF",
        "past": "#F5F7FA",
        "land": "#EEF2F6",
        "ink": "#0F172A",
        "ink2": "#475569",
        "muted": "#64748B",
        "rule": "#CBD5E1",
        "rule2": "#EDF1F5",
        "outturn": "#0F172A",
        "incumbent": "#C2255C",
        "accent": "#1F5FAD",
        "onshore": "#7FAADF",
        "faint": "#94A3B8",
        "fan": ["#DCE7F4", "#C1D4EC", "#A6C2E4", "#8BAFDC"],
    }

    FONT = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'
    CSS = f"""
    .marimo, :root {{ --background: {P["bg"]}; }}
    body {{ background: {P["bg"]}; }}
    .page {{ color: {P["ink"]}; font: 14px/1.5 {FONT}; }}
    .page h1 {{ font-size: 30px; font-weight: 600; letter-spacing: -.02em; line-height: 1.15;
                margin: 4px 0 10px; color: {P["ink"]}; }}
    .page .subhead {{ font-size: 15px; font-weight: 600; margin: 0 0 8px; color: {P["ink"]}; }}
    .page .lede {{ color: {P["ink2"]}; font-size: 15px; max-width: 620px; margin: 0; }}
    .bar {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
            gap: 8px 24px; margin-top: 8px; font-size: 13px; color: {P["muted"]}; }}
    .bar b {{ color: {P["ink"]}; font-weight: 500; }}
    .legend {{ display: flex; gap: 20px; color: {P["ink2"]}; font-size: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
    .legend i {{ display: inline-block; width: 16px; height: 2px; border-radius: 1px; }}
    .legend i.fan {{ width: 18px; height: 12px; border-radius: 2px;
                     background: linear-gradient({P["fan"][0]} 12.5%, {P["fan"][1]} 12.5% 25%, {P["fan"][2]} 25% 37.5%,
                                                 {P["fan"][3]} 37.5% 45%, {P["accent"]} 45% 55%, {P["fan"][3]} 55% 62.5%,
                                                 {P["fan"][2]} 62.5% 75%, {P["fan"][1]} 75% 87.5%, {P["fan"][0]} 87.5%); }}
    .chart {{ display: block; width: 100%; height: auto; margin-top: 14px;
              font: 11px {FONT}; font-variant-numeric: tabular-nums; }}
    .chart text {{ fill: {P["faint"]}; }}
    .chart text.date {{ fill: {P["ink2"]}; font-weight: 500; }}
    .chart text.axis {{ fill: {P["muted"]}; }}
    .chart text.now {{ fill: {P["muted"]}; }}
    .chart text.label {{ fill: {P["muted"]}; font-size: 10.5px; }}
    .chart text.median {{ fill: {P["ink"]}; font-weight: 600; font-size: 10.5px; }}
    @media (max-width: 720px) {{ .chart text {{ font-size: 19px !important; }} }}
    .chart .hover .show {{ visibility: hidden; }}
    .chart .hover:hover .show {{ visibility: visible; }}
    .chart .tip rect {{ fill: {P["bg"]}; stroke: {P["rule"]}; filter: drop-shadow(0 2px 6px rgb(15 23 42 / .10)); }}
    .chart .tip text {{ font-size: 12px; fill: {P["muted"]}; }}
    .chart .tip .title, .chart .tip .value {{ fill: {P["ink"]}; font-weight: 500; }}
    .chart .tip .title {{ font-weight: 600; }}
    .chart .tip .note {{ font-size: 10.5px; }}
    .inputs {{ display: flex; flex-wrap: wrap; gap: 24px 48px; align-items: flex-start; margin-top: 36px;
               padding-top: 24px; border-top: 1px solid {P["rule2"]}; }}
    .inputs .text {{ flex: 1 1 320px; max-width: 560px; color: {P["ink2"]}; font-size: 14px; }}
    .inputs ol {{ margin: 0 0 16px; padding-left: 20px; list-style: decimal; }}
    .inputs li::marker {{ color: {P["faint"]}; }}
    .inputs li {{ margin-bottom: 6px; }}
    .inputs b {{ color: {P["ink"]}; font-weight: 600; }}
    .inputs .map text {{ font: 11px {FONT}; fill: {P["muted"]}; }}
    .scores {{ flex: 0 1 560px; width: 100%; height: auto; font: 12px {FONT}; font-variant-numeric: tabular-nums; }}
    .scores text {{ fill: {P["ink2"]}; }}
    .scores text.head {{ fill: {P["ink"]}; font-weight: 600; }}
    .scores text.ours {{ fill: {P["ink"]}; font-weight: 600; }}
    .scores text.value {{ fill: {P["ink"]}; }}
    .scores text.note {{ fill: {P["muted"]}; font-size: 10.5px; }}
    .inputs .text p + p {{ margin-top: 10px; }}
    .credit {{ color: {P["muted"]}; font-size: 12px; margin-top: 12px; }}
    """
    return CSS, P


@app.cell(hide_code=True)
def _(P, np, pd):
    GW = 1000
    FAN = ((10, 90), (20, 80), (30, 70), (40, 60))
    LABELLED = (90, 70, 50, 30, 10)

    def gw(value) -> str:
        return "–" if pd.isna(value) else f"{value / GW:.1f}"

    def runs(series: pd.Series) -> list[pd.Series]:
        """Unbroken hourly stretches of `series`, so gaps are never bridged."""
        series = series.dropna()
        breaks = (series.index.to_series().diff() != pd.Timedelta("1h")).cumsum()
        return [part for _, part in series.groupby(breaks)]

    def tooltip(t, x, forecast, incumbent, observed, top: float, W: int) -> str:
        """A card of the hour's values, beside the crosshair, flipped left near the right edge."""
        rows, notes = [], []
        if t in observed.index:
            rows.append(("Outturn", f"{gw(observed[t])} GW"))
        if pd.notna(forecast.q50.get(t)):
            row = forecast.loc[t]
            rows += [("Median", f"{gw(row.q50)} GW"), ("p10–p90", f"{gw(row.q10)}–{gw(row.q90)} GW"),
                     ("NESO", f"{gw(incumbent.generation[t])} GW")]
            notes.append(f"Both from NESO's {incumbent.published[t]:%a %H:%M} UTC update")
        width, height = 200, 34 + 18 * len(rows) + 15 * len(notes) + (4 if notes else 0)
        x0 = x + 12 if x + 12 + width < W else x - 12 - width
        out = [f'<rect x="{x0:.1f}" y="{top}" width="{width}" height="{height}" rx="6"/>',
               f'<text x="{x0 + 12:.1f}" y="{top + 21}" class="title">{t:%a %d %b, %H:%M}</text>']
        for k, (label, value) in enumerate(rows):
            y = top + 41 + 18 * k
            out.append(f'<text x="{x0 + 12:.1f}" y="{y}">{label}</text>')
            out.append(f'<text x="{x0 + width - 12:.1f}" y="{y}" text-anchor="end" class="value">{value}</text>')
        for k, note in enumerate(notes):
            out.append(f'<text x="{x0 + 12:.1f}" y="{top + 45 + 18 * len(rows) + 15 * k}" class="note">{note}</text>')
        return f'<g class="tip">{"".join(out)}</g>'

    def spread(ys: list[float], gap: float = 12) -> list[float]:
        """Label positions at `ys` (top to bottom), pushed apart to at least `gap`, outward from the middle."""
        ys, middle = list(ys), len(ys) // 2
        for k in range(middle - 1, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - gap)
        for k in range(middle + 1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + gap)
        return ys

    def chart(forecast: pd.DataFrame, incumbent: pd.DataFrame, observed: pd.Series, now: pd.Timestamp,
              W: int = 960, H: int = 380) -> str:
        """Outturn since yesterday under the forecasts from TabPFN-3.5 (its full distribution) and
        NESO, each hour from the last update issued before it, to the end of tomorrow."""
        left, right, top, bottom = 34, 48, 30, 46
        base = H - bottom
        start = now.normalize() - pd.Timedelta(days=1)
        issued = forecast.dropna(subset=["q50"])
        end = start + pd.Timedelta(hours=71)
        if len(issued):
            end = min(end, issued.index[-1])
        hours = pd.date_range(start, end, freq="h")
        forecast = issued.reindex(hours)
        incumbent = incumbent.reindex(hours)
        observed = observed[start:]
        step = (W - left - right) / (len(hours) - 1)

        def X(t) -> float:
            return left + (t - start) / pd.Timedelta("1h") * step

        def Xs(index) -> np.ndarray:
            return left + ((index - start) / pd.Timedelta("1h")).to_numpy() * step

        peak = np.nanmax([forecast.q90.max(), incumbent.generation.max(), observed.max(), 1])
        ceiling = max(10 * GW, np.ceil(peak / (5 * GW)) * 5 * GW)

        def Y(v):
            return base - np.asarray(v, dtype=float) / ceiling * (base - top)

        def points(series):
            return [f"{x:.1f},{y:.1f}" for x, y in zip(Xs(series.index), Y(series))]

        def line(series, colour, width):
            d = "".join("M" + "L".join(points(part)) for part in runs(series))
            return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>')

        out = [f'<rect x="{left}" y="{top}" width="{min(X(now), W - right) - left:.1f}" '
               f'height="{base - top}" fill="{P["past"]}"/>']
        for v in np.arange(0, ceiling + 1, 5 * GW):
            out.append(f'<line x1="{left}" x2="{W - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                       f'stroke="{P["rule"] if v == 0 else P["rule2"]}"/>')
            out.append(f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v / GW:.0f}</text>')
        out.append(f'<text x="0" y="{top - 14}" class="axis">Wind output, GW</text>')
        for t in hours[(hours.hour % 6 == 0) & (hours.hour > 0)]:
            out.append(f'<text x="{X(t):.1f}" y="{base + 16}" text-anchor="middle">{t:%H:%M}</text>')
        for day in pd.date_range(start, end.normalize()):
            x = X(day)
            if day != start:
                out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{base + 40}" stroke="{P["rule2"]}"/>')
            close = X(min(day + pd.Timedelta(days=1), end))
            if close - x > 12 * step:
                out.append(f'<text x="{(x + close) / 2:.1f}" y="{base + 36}" text-anchor="middle" '
                           f'class="date">{day:%A %d %b}</text>')

        for part in runs(forecast.q50):
            block = forecast.loc[part.index]
            for (lower, upper), colour in zip(FAN, P["fan"]):
                edge = points(block[f"q{lower}"]) + points(block[f"q{upper}"])[::-1]
                out.append(f'<path d="M{"L".join(edge)}Z" fill="{colour}"/>')
        out.append(line(forecast.q50, P["accent"], 2))
        out.append(line(incumbent.generation, P["incumbent"], 1.5))
        out.append(line(observed, P["outturn"], 2))

        x = X(now)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{base}" stroke="{P["faint"]}" '
                   f'stroke-dasharray="4 3"/>')
        out.append(f'<text x="{x:.1f}" y="{top - 9}" text-anchor="middle" class="now">Now</text>')
        if len(observed):
            out.append(f'<circle cx="{X(observed.index[-1]):.1f}" cy="{Y(observed.iloc[-1]):.1f}" r="4" '
                       f'fill="{P["outturn"]}" stroke="{P["bg"]}" stroke-width="2"/>')

        if forecast.q50.notna().any():
            last = forecast.q50.last_valid_index()
            edges = [float(Y(forecast.at[last, f"q{p}"])) for p in LABELLED]
            for p, edge, y in zip(LABELLED, edges, spread(edges)):
                out.append(f'<path d="M{X(last) + 3:.1f},{edge:.1f}H{X(last) + 9:.1f}L{X(last) + 14:.1f},{y:.1f}" '
                           f'fill="none" stroke="{P["rule"]}"/>')
                out.append(f'<text x="{X(last) + 17:.1f}" y="{y + 3.5:.1f}" '
                           f'class="{"median" if p == 50 else "label"}">p{p}</text>')

        for t in hours:
            x = X(t)
            out.append(
                f'<g class="hover"><rect x="{x - step / 2:.1f}" y="{top}" width="{step:.1f}" '
                f'height="{base - top}" fill="transparent"/>'
                f'<g class="show"><line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{base}" '
                f'stroke="{P["ink2"]}" stroke-opacity=".4"/>{tooltip(t, x, forecast, incumbent, observed, top + 6, W)}</g></g>'
            )
        return f'<svg class="chart" viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'

    def sites(meta, W: int = 230) -> str:
        """GB wind farms, and the 20 capacity-weighted points whose weather TabPFN-3.5 reads."""
        west, east, south, north = -8.2, 2.3, 49.9, 59.1
        squash = np.cos(np.radians((south + north) / 2))
        scale = W / ((east - west) * squash)
        H = round((north - south) * scale)

        def xy(lon, lat):
            return (lon - west) * squash * scale, (north - lat) * scale

        out = []
        for ring in meta.outline:
            d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in (xy(lon, lat) for lon, lat in ring)) + "Z"
            out.append(f'<path d="{d}" fill="{P["land"]}" stroke="{P["rule"]}" stroke-width=".6"/>')
        for lon, lat, mw in meta.farms:
            x, y = xy(lon, lat)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{np.sqrt(mw) / 9:.2f}" fill="{P["faint"]}" fill-opacity=".55"/>')
        for lon, lat, mw, offshore in sorted(meta.points, key=lambda point: -point[2]):
            x, y = xy(lon, lat)
            colour = P["accent"] if offshore else P["onshore"]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{np.sqrt(mw) / 6:.1f}" fill="{colour}" '
                       f'stroke="{P["bg"]}" stroke-width="1.5"/>')
        key = [(P["accent"], "offshore point"), (P["onshore"], "onshore point")]
        for k, (colour, label) in enumerate(key):
            out.append(f'<circle cx="{W - 96}" cy="{14 + 18 * k}" r="5" fill="{colour}"/>')
            out.append(f'<text x="{W - 86}" y="{18 + 18 * k}">{label}</text>')
        out.append(f'<circle cx="{W - 96}" cy="{50}" r="2" fill="{P["faint"]}"/>')
        out.append(f'<text x="{W - 86}" y="{54}">wind farm</text>')
        return f'<svg class="map" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'

    def scores(meta, W: int = 560) -> str:
        """Held-out mean error and p10–p90 coverage, TabPFN-3.5 against NESO and the two baselines."""
        label, panel, gap, row, top = 150, 170, 40, 30, 30
        mae_x, cover_x = label, label + panel + gap
        bottom = top + row * len(meta.scores)
        H = bottom + 16
        colours = {"NESO": P["incumbent"], "NESO + conformal band": P["incumbent"]}
        out = [f'<text x="{mae_x}" y="12" class="head">Mean error, MW</text>',
               f'<text x="{cover_x}" y="12" class="head">Hours inside p10–p90</text>']
        target = cover_x + 0.8 * panel
        out.append(f'<line x1="{target:.1f}" x2="{target:.1f}" y1="{top}" y2="{bottom}" stroke="{P["ink2"]}" '
                   f'stroke-dasharray="3 3"/>')
        out.append(f'<text x="{target:.1f}" y="{H - 2}" text-anchor="middle" class="note">80% target</text>')
        for k, (name, mae, cover) in enumerate(meta.scores):
            ours = name.startswith("TabPFN")
            y, colour = top + row * k, P["accent"] if ours else colours.get(name, P["faint"])
            weight = ' class="ours"' if ours else ""
            out.append(f'<text x="0" y="{y + 15}"{weight}>{name}</text>')
            width = mae / 1200 * panel
            out.append(f'<rect x="{mae_x}" y="{y + 4}" width="{width:.1f}" height="14" rx="2" fill="{colour}"/>')
            out.append(f'<text x="{mae_x + width + 6:.1f}" y="{y + 15}" class="value">{mae:,}</text>')
            if cover is None:
                out.append(f'<text x="{cover_x}" y="{y + 15}" class="note">point forecast only</text>')
                continue
            width = cover * panel
            out.append(f'<rect x="{cover_x}" y="{y + 4}" width="{width:.1f}" height="14" rx="2" fill="{colour}"/>')
            out.append(f'<text x="{W}" y="{y + 15}" text-anchor="end" class="value">{cover:.0%}</text>')
        return f'<svg class="scores" viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'

    return chart, scores, sites


@app.cell(hide_code=True)
def _(CSS, mo):
    mo.Html(f"""
    <style>{CSS}</style>
    <header class="page">
      <h1>Great Britain wind power</h1>
      <p class="lede">TabPFN-3.5's forecast of wind output as a full probability distribution,
      against the grid operator's own. Past hours show the last forecast issued before each,
      so outturn can be read against both.</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(P, chart, incumbent, live, mo, newest, now, observed, pd):
    _observed = observed if observed is not None else pd.Series(dtype=float)
    _parts = [f"Latest forecast issued <b>{live.issue_time.max():%a %d %b, %H:%M} UTC</b>"]
    if newest is not None and newest > incumbent.published.max():
        _parts.append(f"NESO's {newest:%H:%M} update not yet included")
    if observed is None:
        _parts.append("live outturn unavailable")
    mo.Html(f"""
    <div class="page">
      <div class="bar">
        <span>{" · ".join(_parts)}</span>
        <span class="legend">
          <span><i style="background:{P["outturn"]}"></i>Outturn</span>
          <span><i class="fan"></i>TabPFN-3.5</span>
          <span><i style="background:{P["incumbent"]}"></i>NESO</span>
        </span>
      </div>
      {chart(live, incumbent, _observed, now)}
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(meta, mo, scores):
    _scores = {label: (mae, cover) for label, mae, cover in meta.scores}
    _neso, _ifs, _aifs, _lgbm = (_scores[k] for k in ("NESO", "TabPFN-3.5 · IFS", "TabPFN-3.5 · AIFS", "LightGBM · AIFS"))
    mo.Html(f"""
    <div class="page inputs">
      {scores(meta)}
      <div class="text">
        <p class="subhead">Held out: {meta.days} days, 2025–26</p>
        <p>NESO's forecast missed by {_neso[0]:,} MW on average. TabPFN-3.5's median missed by
        <b>{_ifs[0]:,} MW</b> on ECMWF IFS weather and <b>{_aifs[0]:,} MW</b> on AIFS, the archive
        this page runs on: {1 - _ifs[0] / _neso[0]:.0%} and {1 - _aifs[0] / _neso[0]:.0%} less. Its
        p10–p90 held {_ifs[1]:.0%} and {_aifs[1]:.0%} of hours with no calibration step. A LightGBM
        tuned on the same AIFS inputs missed by {_lgbm[0]:,} MW, and its p10–p90 held {_lgbm[1]:.0%}.</p>
        <p>What the day-ahead forecast misses is made good in real time, from reserve plant or by
        turning generation down. NESO spent £2.3bn balancing Britain's grid in 2025, and its wind
        forecast missed by 9.4 TWh. Priced at the gap between imbalance and market prices, each
        tenth of that miss costs about £20m a year, and it grows with every gigawatt of wind built.</p>
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(meta, mo, sites):
    mo.Html(f"""
    <div class="page inputs">
      {sites(meta)}
      <div class="text">
        <p class="subhead">What each forecast reads</p>
        <ol>
          <li>NESO's forecast, fifteen minutes after each of its updates.</li>
          <li>ECMWF AIFS's newest 10 m wind at the <b>20 points</b> on the map, each the
          capacity-weighted centre of the wind farms around it.</li>
          <li>Every settled hour of outturn since April 2024, as context. TabPFN-3.5 is not
          trained or tuned for this: each forecast is one forward pass.</li>
        </ol>
        <p class="credit">Outturn is metered transmission wind plus balancing-mechanism curtailment,
        hour ending · Data: Elexon BMRS, ECMWF AIFS via dynamical.org (CC BY 4.0), REPD (OGL v3.0) ·
        Protocol frozen at {meta.freeze}</p>
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

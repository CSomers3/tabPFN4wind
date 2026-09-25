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
    import json
    import sys
    import urllib.request
    from urllib.parse import urlencode

    import marimo as mo
    import numpy as np
    import pandas as pd

    return asyncio, json, mo, np, pd, sys, urlencode, urllib


@app.cell(hide_code=True)
def _(mo, pd):
    # Written by `windpfn-page` and `windpfn-live`: windpfn itself can't be installed under Pyodide.
    public = mo.notebook_location() / "public"
    meta = pd.read_json(str(public / "meta.json"), typ="series")
    live = pd.read_csv(str(public / "live.csv"), index_col="hour", parse_dates=["hour", "issue_time"])
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
def _(np):
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
    }
    TAIL, CENTRE = np.array([228, 237, 249]), np.array([62, 122, 196])

    def shade(depth: float) -> str:
        """Blue for a percentile `depth` from the median: 0 at p0/p100, 1 at p50."""
        red, green, blue = (TAIL + (CENTRE - TAIL) * depth).round().astype(int)
        return f"#{red:02X}{green:02X}{blue:02X}"

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
                     background: linear-gradient({shade(0.1)}, {shade(0.9)} 42%, #fff 42% 58%, {shade(0.9)} 58%, {shade(0.1)}); }}
    .chart {{ display: block; width: 100%; height: auto; margin-top: 14px;
              font: 11px {FONT}; font-variant-numeric: tabular-nums; }}
    .chart text {{ fill: {P["faint"]}; }}
    .chart text.date {{ fill: {P["ink"]}; font-weight: 600; }}
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
    .credit {{ color: {P["muted"]}; font-size: 12px; margin-top: 12px; }}
    """
    return CSS, P, shade


@app.cell(hide_code=True)
def _(P, np, pd, shade):
    GW = 1000
    LABELLED = (99, 90, 50, 10, 1)

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
        """Outturn since yesterday, then the most recent forecasts from TabPFN-3.5 (its full
        distribution) and NESO, from the last settled hour to the end of tomorrow."""
        left, right, top, bottom = 34, 48, 26, 28
        base = H - bottom
        start = now.normalize() - pd.Timedelta(days=1)
        joined = observed.index[-1] if len(observed) else now.floor("h")
        ahead = forecast.loc[joined:, :].dropna(subset=["q50"])
        end = start + pd.Timedelta(hours=71)
        if len(ahead):
            end = min(end, ahead.index[-1])
        hours = pd.date_range(start, end, freq="h")
        forecast = ahead.reindex(hours)
        incumbent = incumbent[joined:].reindex(hours)
        observed = observed[start:]
        step = (W - left - right) / (len(hours) - 1)

        def X(t) -> float:
            return left + (t - start) / pd.Timedelta("1h") * step

        def Xs(index) -> np.ndarray:
            return left + ((index - start) / pd.Timedelta("1h")).to_numpy() * step

        peak = np.nanmax([forecast.q99.max(), incumbent.generation.max(), observed.max(), 1])
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
        for t in hours[hours.hour % 6 == 0]:
            x = X(t)
            if t.hour == 0:
                if t != start:
                    out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{base}" stroke="{P["rule2"]}"/>')
                anchor = "start" if t == start else "middle"
                out.append(f'<text x="{x:.1f}" y="{base + 17}" text-anchor="{anchor}" class="date">{t:%a %d %b}</text>')
            else:
                out.append(f'<text x="{x:.1f}" y="{base + 17}" text-anchor="middle">{t:%H:%M}</text>')

        for part in runs(forecast.q50):
            block = forecast.loc[part.index]
            for lower in range(1, 50):
                edge = points(block[f"q{lower}"]) + points(block[f"q{100 - lower}"])[::-1]
                out.append(f'<path d="M{"L".join(edge)}Z" fill="{shade(lower / 50)}"/>')
        out.append(line(forecast.q50, P["bg"], 1.5))
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

    return chart, sites


@app.cell(hide_code=True)
def _(CSS, mo):
    mo.Html(f"""
    <style>{CSS}</style>
    <header class="page">
      <h1>Great Britain wind power</h1>
      <p class="lede">TabPFN-3.5's forecast of tomorrow's output as a full probability
      distribution, against the grid operator's own.</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(P, chart, incumbent, live, mo, newest, now, observed, pd):
    _observed = observed if observed is not None else pd.Series(dtype=float)
    _parts = ["Live outturn unavailable" if observed is None else
              f"Outturn to <b>{_observed.index[-1]:%H:%M}</b>" if len(_observed) else "No outturn yet"]
    _read = incumbent.published.max()
    _parts.append(f"forecasts from NESO's <b>{_read:%a %H:%M}</b> update")
    if newest is not None and newest > _read:
        _parts.append(f"NESO updated at {newest:%H:%M}; TabPFN-3.5 follows 15 minutes after")
    mo.Html(f"""
    <div class="page">
      <div class="bar">
        <span>{" · ".join(_parts)} · UTC, GW</span>
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
          <li>Every settled hour of outturn since March 2024, read in context: TabPFN-3.5 is
          not trained or tuned for this.</li>
        </ol>
        <p>Over <b>{meta.days} held-out days</b> of day-ahead forecasts, TabPFN-3.5's median
        missed by <b>{meta.mae:,.0f} MW</b> on average, against {meta.mae_windfor:,.0f} MW for
        NESO's, and <b>{meta.cover80:.0%}</b> of hours fell between its p10 and p90.</p>
        <p class="credit">Outturn is metered transmission wind plus balancing-mechanism curtailment,
        hour ending · Data: Elexon BMRS, ECMWF AIFS via dynamical.org (CC BY 4.0), REPD (OGL v3.0) ·
        Protocol frozen at {meta.freeze}</p>
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

"""Page data for docs/diagnostics from the held-out forecasts saved by 03_test.ipynb."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from windpfn import dataset, model

ROOT = Path(__file__).resolve().parents[1]
UP, WX = "tabpfn+windfor", "tabpfn"


def debiased(f):
    """WINDFOR minus its trailing 28-day mean error, using only days settled a day before issue."""
    h = dataset.build("2024-11-25", f.index.max().strftime("%Y-%m-%d")).set_index("valid_time")
    b = (h.windfor - h.y).resample("D").mean().rolling(28).mean().shift(3)
    return f.windfor - b.reindex(f.index.floor("D")).to_numpy()


def main():
    f = pd.read_parquet(ROOT / "data/test_forecasts.parquet")
    p = {k: f[f"{k}|q50"] for k in (WX, f"{WX} frozen", UP, f"{UP} frozen")}
    err = pd.DataFrame({"windfor": f.windfor, "windfor debiased": debiased(f), "blend": f.blend, **p}).sub(f.y, axis=0)
    ae, lvl = err.abs(), pd.cut(f.windfor, [0, 3e3, 6e3, 9e3, 12e3, 15e3, 18e3, 25e3])
    day = ae.groupby(ae.index.floor("D")).mean()
    top = day.nlargest(12, "windfor")
    mon = ae.resample("MS").mean().rename(index=lambda t: t.strftime("%Y-%m"))
    lv = ae.groupby(lvl, observed=True).mean().rename(index=lambda i: f"{i.left / 1e3:g}–{i.right / 1e3:g}")
    tab = lambda s: {"index": [str(i) for i in s.index], **{k: s[k].round().tolist() for k in s}}
    data = {
        "commit": subprocess.run(["git", "log", "--format=%h", "--diff-filter=A", "--", "03_test.ipynb"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "window": [f.index.min().isoformat(), f.index.max().isoformat()], "hours": len(f), "days": len(day),
        "score": model.score(err).reset_index(names="model").to_dict("records"),
        "better": float((ae[UP] < ae.windfor).mean()), "better_days": float((day[UP] < day.windfor).mean()),
        "monthly": tab(mon), "lead": tab(ae.groupby(f.lead_h.round().astype(int)).mean()),
        "level": {**tab(lv), "n": f.groupby(lvl, observed=True).size().tolist()},
        "reliability": {k: [float((f.y < f[f"{k}|q{q}"]).mean()) for q in range(10, 100, 10)] for k in (WX, UP)},
        "coverage": {k: float(f.y.between(f[f"{k}|q10"], f[f"{k}|q90"]).mean()) for k in (WX, UP)},
        "worst": [{"day": d.strftime("%Y-%m-%d"), "windfor": round(r.windfor), "up": round(r[UP]), "wx": round(r[WX])} for d, r in top.iterrows()],
        "hourly": {"t": (f.index.astype("int64") // 3_600_000_000_000).tolist(),
                   **{k: f[c].round().astype(int).tolist() for k, c in
                      [("y", "y"), ("windfor", "windfor"), ("wx", f"{WX}|q50"), ("up", f"{UP}|q50"), ("lo", f"{UP}|q10"), ("hi", f"{UP}|q90")]}},
    }
    out = ROOT / "docs/diagnostics/data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()

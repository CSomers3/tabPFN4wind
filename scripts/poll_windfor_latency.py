"""Poll WINDFOR and log when each new publishTime first becomes visible (latency audit)."""
import json, time, sys
from datetime import datetime, timezone, timedelta
import requests
B = "https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR/stream"
out = "data/audit/windfor_first_seen.jsonl"
seen = set(); stop = datetime.now(timezone.utc) + timedelta(hours=float(sys.argv[1]) if len(sys.argv) > 1 else 3)
while datetime.now(timezone.utc) < stop:
    now = datetime.now(timezone.utc)
    try:
        r = requests.get(B, params=dict(publishDateTimeFrom=(now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%MZ"),
                                        publishDateTimeTo=(now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%MZ")), timeout=60)
        pts = {x["publishTime"] for x in r.json()}
    except Exception as e:
        pts = set(); print("err", e, flush=True)
    for p in sorted(pts - seen):
        rec = {"publishTime": p, "first_seen_utc": now.isoformat(timespec="seconds"), "initial": not seen}
        open(out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
    seen |= pts
    time.sleep(120)

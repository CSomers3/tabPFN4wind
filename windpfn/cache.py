"""Content-addressed cache for raw HTTP pulls.

Every response is stored under data/raw/<source>/<key>.json, where key is a hash of
the request URL + sorted params. A manifest (data/raw/manifest.jsonl) records the
full URL, fetch time and sha256 of the stored bytes so pulls are auditable.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

DATA = Path(__file__).resolve().parents[1] / "data"
ROOT = DATA / "raw"
MANIFEST = ROOT / "manifest.jsonl"


def _key(url: str, params: dict | None) -> tuple[str, str]:
    full = url + ("?" + urlencode(sorted((params or {}).items())) if params else "")
    return hashlib.sha256(full.encode()).hexdigest()[:24], full


def get_json(source: str, url: str, params: dict | None = None, *, refresh: bool = False,
             retries: int = 4, timeout: int = 120):
    key, full = _key(url, params)
    path = ROOT / source / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_bytes())
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            r = None
        if r is not None and r.status_code != 429 and r.status_code < 500:
            break
        if attempt == retries - 1:
            break
        time.sleep(2 ** attempt * 2)
    if r is None:
        raise requests.ConnectionError(full)
    r.raise_for_status()
    body = r.content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    with MANIFEST.open("a") as f:
        f.write(json.dumps({"source": source, "key": key, "url": full,
                            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}) + "\n")
    return json.loads(body)

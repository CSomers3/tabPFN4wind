import argparse
from . import data, nwp

p = argparse.ArgumentParser(prog="windpfn")
p.add_argument("cmd", choices=["fetch"])
p.add_argument("--start", default="2024-03-16")
a = p.parse_args()
if a.cmd == "fetch":
    data.fuelhh(a.start)
    nwp.backfill(data.issues(a.start))

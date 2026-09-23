"""Blog/paper figures for the 2024 exploration window -> figures/*.png, *.pdf."""
import matplotlib
matplotlib.use("Agg")
from windpfn import baselines, data, dataset, figures, sites
from windpfn.cache import DATA

A, B = "2024-03-16", "2024-12-31"
d = dataset.build(A, B).set_index("valid_time")
d = baselines.add(d.assign(price=dataset.hourly(data.prices("2024-03-09", B)).reindex(d.index)))
print(figures.save_all(d, *sites.clusters(), out=DATA.parent / "figures"))

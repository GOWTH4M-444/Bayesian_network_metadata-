"""Pick the network density (structure penalty, max parents) that predicts best on held-out cells."""
from __future__ import annotations
import pandas as pd
from .model import MetadataBN
from .evaluate import evaluate_observed

DEFAULT_GRID = [(1.0, 3), (0.5, 3), (0.25, 3), (0.5, 4), (0.25, 4), (0.1, 4)]


def auto_tune(df: pd.DataFrame, cols: list[str], min_count: int = 20, grid=None, sample: int = 15000,
              val_frac: float = 0.2, em_iters: int = 2, seed: int = 0):
    """Returns (best_penalty, best_max_parents, table). Lower penalty = more edges (richer network)."""
    d = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
    nv = int(len(d) * val_frac)
    val, tr = d.iloc[:nv], d.iloc[nv:]
    rows = []
    for pen, mp in (grid or DEFAULT_GRID):
        m = MetadataBN(min_count=min_count, max_parents=mp, penalty=pen, em_iters=em_iters).fit(tr, cols)
        r = evaluate_observed(m, val, tr, max_cells=300)
        rows.append({"penalty": pen, "max_parents": mp, "n_edges": len(m.edges()),
                     "mean_top1_accuracy": float(r["BN top-1"].mean())})
    table = pd.DataFrame(rows)
    best = table.loc[table.mean_top1_accuracy.idxmax()]
    return float(best.penalty), int(best.max_parents), table

"""Edge-confidence by bootstrap + network diagram."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from .data import MetadataCodec
from .structure import learn_structure


def bootstrap_edges(df: pd.DataFrame, cols: list[str], min_count: int = 20, max_parents: int = 3,
                    n_boot: int = 15, sample: int = 15000, seed: int = 0, penalty: float = 1.0) -> pd.DataFrame:
    """Re-learn the structure on resampled studies; report how often each edge reappears."""
    codec = MetadataCodec(min_count=min_count).fit(df, cols)
    X = codec.transform(df)
    rng = np.random.default_rng(seed)
    directed, skeleton = Counter(), Counter()
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), min(sample, len(X)))
        parents = learn_structure(X[idx], codec.card, max_parents, penalty=penalty)
        for c, ps in parents.items():
            for p in ps:
                directed[(codec.columns[p], codec.columns[c])] += 1
                skeleton[frozenset((codec.columns[p], codec.columns[c]))] += 1
    rows = [{"parent": p, "child": c, "same_direction": n / n_boot,
             "any_direction": skeleton[frozenset((p, c))] / n_boot} for (p, c), n in directed.items()]
    return pd.DataFrame(rows).sort_values("any_direction", ascending=False).reset_index(drop=True)


def edge_confidence_table(model, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, c in model.edges():
        m = boot[(boot.parent == p) & (boot.child == c)]
        sk = boot[((boot.parent == p) & (boot.child == c)) | ((boot.parent == c) & (boot.child == p))]
        rows.append({"edge": f"{p} -> {c}",
                     "found_in_resamples (any direction)": float(sk.any_direction.max()) if len(sk) else 0.0,
                     "found_in_this_direction": float(m.same_direction.iloc[0]) if len(m) else 0.0})
    return pd.DataFrame(rows).set_index("edge")


def plot_network(model, boot: pd.DataFrame | None, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    names = model.net.names
    n = len(names)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {nm: (np.cos(a), np.sin(a)) for nm, a in zip(names, ang)}
    conf = edge_confidence_table(model, boot) if boot is not None else None
    fig, ax = plt.subplots(figsize=(11, 9))
    for p, c in model.edges():
        (x1, y1), (x2, y2) = pos[p], pos[c]
        w = 1.0
        if conf is not None:
            w = 0.5 + 5 * conf.loc[f"{p} -> {c}", "found_in_resamples (any direction)"]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=w,
                                     color="#3a6ea5", alpha=0.75, shrinkA=22, shrinkB=22))
    for nm, (x, y) in pos.items():
        ax.text(x, y, nm.replace("_", "\n"), ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f9", ec="#3a6ea5"))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.4, 1.4); ax.axis("off")
    ax.set_title("Learned metadata network (arrow width = how often the link reappears in resampled data)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

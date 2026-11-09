"""Evaluation: held-out imputation accuracy vs. a marginal baseline, and structure recovery."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .model import MetadataBN


def imputation_accuracy(model: MetadataBN, df_complete: pd.DataFrame, df_train: pd.DataFrame, top_k: int = 3):
    """Hide each field in turn in fully-observed held-out studies and predict it from the rest."""
    X = model.codec.transform(df_complete)
    ok = (X >= 0).all(axis=1)
    X = X[ok]
    Xtr = model.codec.transform(df_train)
    rows = []
    for j, c in enumerate(model.codec.columns):
        vals = Xtr[:, j][Xtr[:, j] >= 0]
        majority = np.bincount(vals, minlength=model.net.card[j]).argmax()
        hit1 = hitk = hit_base = 0
        for r in X:
            ev = {k: int(s) for k, s in enumerate(r) if k != j}
            p = model.net.predict_proba(j, ev)
            order = np.argsort(-p)
            hit1 += order[0] == r[j]
            hitk += r[j] in order[:top_k]
            hit_base += majority == r[j]
        n = len(X)
        rows.append({"field": c, "BN top-1": hit1 / n, f"BN top-{top_k}": hitk / n, "baseline (majority)": hit_base / n})
    return pd.DataFrame(rows).set_index("field")


def structure_recovery(learned: set[tuple[str, str]], truth: set[tuple[str, str]]):
    """Edge precision/recall (directed and skeleton). Directions within a Markov equivalence
    class are not identifiable from observational data, so the skeleton metric is the fairer one."""
    def prf(a, b):
        tp = len(a & b)
        p = tp / len(a) if a else 0.0
        r = tp / len(b) if b else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    skel = lambda s: {frozenset(e) for e in s}
    return {"directed (P,R,F1)": prf(set(learned), set(truth)),
            "skeleton (P,R,F1)": prf(skel(learned), skel(truth))}


def evaluate_observed(model: MetadataBN, df_test: pd.DataFrame, df_train: pd.DataFrame,
                      top_k: int = 3, max_cells: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Works on real, incomplete data: for each field, hide every *observed* value of the held-out
    studies in turn and predict it from that study's other observed fields."""
    rng = np.random.default_rng(seed)
    X = model.codec.transform(df_test)
    Xtr = model.codec.transform(df_train)
    rows = []
    for j, c in enumerate(model.codec.columns):
        vals = Xtr[:, j][Xtr[:, j] >= 0]
        majority = np.bincount(vals, minlength=model.net.card[j]).argmax()
        idx = np.where(X[:, j] >= 0)[0]
        if len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        h1 = hk = hb = 0
        for i in idx:
            ev = {k: int(s) for k, s in enumerate(X[i]) if s >= 0 and k != j}
            p = model.net.predict_proba(j, ev)
            order = np.argsort(-p)
            h1 += order[0] == X[i, j]
            hk += X[i, j] in order[:top_k]
            hb += majority == X[i, j]
        n = max(len(idx), 1)
        rows.append({"field": c, "n_tested": len(idx), "BN top-1": h1 / n,
                     f"BN top-{top_k}": hk / n, "baseline (majority)": hb / n})
    return pd.DataFrame(rows).set_index("field")

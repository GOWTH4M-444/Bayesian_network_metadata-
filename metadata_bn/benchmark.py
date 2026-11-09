"""Comparison against a standard classifier (random forest) on exactly the same task."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_random_forest(model, df_test: pd.DataFrame, df_train: pd.DataFrame, top_k: int = 3,
                           max_cells: int = 1000, max_train: int = 20000, seed: int = 0) -> pd.DataFrame:
    """For each field: train a random forest on the other fields (missing = its own category) using studies
    where the field is observed, then test it on the same kind of held-out cells as the Bayesian network."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder

    Xte = model.codec.transform(df_test)
    Xtr = model.codec.transform(df_train)
    rng = np.random.default_rng(seed)
    rows = []
    for j, c in enumerate(model.codec.columns):
        others = [k for k in range(Xtr.shape[1]) if k != j]
        tr = np.where(Xtr[:, j] >= 0)[0]
        if len(tr) > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        te = np.where(Xte[:, j] >= 0)[0]
        if len(te) > max_cells:
            te = rng.choice(te, max_cells, replace=False)
        if len(tr) < 20 or len(te) == 0:
            rows.append({"field": c, "RandomForest top-1": np.nan, f"RandomForest top-{top_k}": np.nan})
            continue
        enc = OneHotEncoder(handle_unknown="ignore")
        A = enc.fit_transform(Xtr[np.ix_(tr, others)] + 1)
        B = enc.transform(Xte[np.ix_(te, others)] + 1)
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        rf.fit(A, Xtr[tr, j])
        proba = rf.predict_proba(B)
        order = np.argsort(-proba, axis=1)[:, :top_k]
        top = rf.classes_[order]
        truth = Xte[te, j]
        rows.append({"field": c, "RandomForest top-1": float((top[:, 0] == truth).mean()),
                     f"RandomForest top-{top_k}": float((top == truth[:, None]).any(axis=1).mean())})
    return pd.DataFrame(rows).set_index("field")

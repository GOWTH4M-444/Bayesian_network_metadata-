"""High-level API: fit on a DataFrame of study metadata, predict missing fields."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .data import MetadataCodec
from .network import BayesianNetwork
from .structure import learn_structure


class MetadataBN:
    def __init__(self, min_count: int = 5, max_parents: int = 3, ess: float = 2.0,
                 em_iters: int = 10, verbose: bool = False, penalty: float = 1.0):
        self.codec = MetadataCodec(min_count=min_count)
        self.max_parents, self.ess, self.em_iters, self.verbose = max_parents, ess, em_iters, verbose
        self.penalty = penalty
        self.net: BayesianNetwork | None = None

    # -------------------------------------------------------------------- train
    def fit(self, df: pd.DataFrame, columns: list[str] | None = None,
            whitelist: list[tuple[str, str]] | None = None,
            blacklist: list[tuple[str, str]] | None = None) -> "MetadataBN":
        """whitelist/blacklist: lists of (parent_name, child_name) edges for domain knowledge."""
        self.codec.fit(df, columns)
        X = self.codec.transform(df)
        cols = self.codec.columns
        ix = {c: i for i, c in enumerate(cols)}
        wl = {(ix[a], ix[b]) for a, b in (whitelist or [])}
        bl = {(ix[a], ix[b]) for a, b in (blacklist or [])}
        if self.verbose:
            print("Learning structure ...")
        parents = learn_structure(X, self.codec.card, self.max_parents, whitelist=wl, blacklist=bl,
                                  verbose=self.verbose, penalty=self.penalty)
        self.net = BayesianNetwork(cols, self.codec.card, parents)
        if self.verbose:
            print("Fitting parameters (EM over incomplete rows) ...")
        self.net.fit(X, self.ess)
        if self.em_iters and (X < 0).any():
            self.net.fit_em(X, self.ess, max_iter=self.em_iters, verbose=self.verbose)
        return self

    # ------------------------------------------------------------------ predict
    def predict_proba(self, record: dict, target: str) -> dict[str, float]:
        ev = self.codec.evidence_from_dict(record)
        t = self.codec.columns.index(target)
        ev.pop(t, None)
        p = self.net.predict_proba(t, ev)
        return dict(zip(self.codec.categories[target], map(float, p)))

    def predict(self, record: dict, top_k: int = 3) -> dict[str, list[tuple[str, float]]]:
        """For every field absent from `record`, return its top-k most probable values."""
        out = {}
        for c in self.codec.columns:
            if c in record and record[c] is not None and not pd.isna(record[c]):
                continue
            probs = self.predict_proba(record, c)
            out[c] = sorted(probs.items(), key=lambda kv: -kv[1])[:top_k]
        return out

    def fill_missing(self, df: pd.DataFrame, min_confidence: float = 0.0) -> pd.DataFrame:
        """Impute the most probable value for each missing cell (with confidence columns)."""
        X = self.codec.transform(df)
        res = self.codec.inverse_transform(X)
        conf = pd.DataFrame(np.nan, index=res.index, columns=[f"{c}__p" for c in res.columns])
        for i in range(len(df)):
            ev = {j: int(s) for j, s in enumerate(X[i]) if s >= 0}
            for j, c in enumerate(self.codec.columns):
                if j in ev:
                    continue
                p = self.net.predict_proba(j, ev)
                k = int(p.argmax())
                if p[k] >= min_confidence:
                    res.iat[i, j] = self.codec.categories[c][k]
                    conf.iat[i, j] = p[k]
        return pd.concat([res, conf], axis=1)

    # -------------------------------------------------------------- introspection
    def edges(self):
        return self.net.edges()

    def markov_blanket(self, field: str) -> set[str]:
        n = self.net
        i = n.names.index(field)
        mb = set(n.parents[i])
        for c in range(n.n):
            if i in n.parents[c]:
                mb.add(c)
                mb |= set(n.parents[c])
        mb.discard(i)
        return {n.names[j] for j in mb}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MetadataBN":
        with open(path, "rb") as f:
            return pickle.load(f)

"""Discrete Bayesian network: parameters, EM for missing data, prediction."""
from __future__ import annotations
import numpy as np
from .inference import query


class BayesianNetwork:
    def __init__(self, names: list[str], card: list[int], parents: dict[int, list[int]] | None = None):
        self.names = list(names)
        self.card = list(card)
        self.n = len(names)
        self.parents = {i: sorted((parents or {}).get(i, [])) for i in range(self.n)}
        self.cpts: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- structure
    def edges(self) -> list[tuple[str, str]]:
        return [(self.names[p], self.names[c]) for c, ps in self.parents.items() for p in ps]

    def _family_shape(self, node):
        return tuple(self.card[p] for p in self.parents[node]) + (self.card[node],)

    def _pseudo(self, node, ess):
        shape = self._family_shape(node)
        return ess / np.prod(shape)  # BDeu-style per-cell pseudo count

    # --------------------------------------------------------------- parameters
    def fit(self, X: np.ndarray, ess: float = 2.0) -> "BayesianNetwork":
        """Available-case MLE with Dirichlet smoothing (rows where the whole family is observed)."""
        for node in range(self.n):
            fam = self.parents[node] + [node]
            sub = X[:, fam]
            sub = sub[(sub >= 0).all(axis=1)]
            counts = np.zeros(self._family_shape(node))
            np.add.at(counts, tuple(sub.T), 1.0)
            self.cpts[node] = self._normalise(counts + self._pseudo(node, ess))
        return self

    def fit_em(self, X: np.ndarray, ess: float = 2.0, max_iter: int = 15, tol: float = 1e-4,
               verbose: bool = False) -> "BayesianNetwork":
        """EM that uses all rows, including those with missing metadata (MAR assumption)."""
        if not self.cpts:
            self.fit(X, ess)
        # group identical rows -> far fewer inference calls
        uniq, w = np.unique(X, axis=0, return_counts=True)
        for it in range(max_iter):
            counts = {n: np.zeros(self._family_shape(n)) for n in range(self.n)}
            for row, wt in zip(uniq, w):
                ev = {j: int(s) for j, s in enumerate(row) if s >= 0}
                for node in range(self.n):
                    fam = self.parents[node] + [node]
                    unobs = [v for v in fam if v not in ev]
                    if not unobs:
                        counts[node][tuple(ev[v] for v in fam)] += wt
                    else:
                        post = query(self, unobs, ev)
                        sl = tuple(ev[v] if v in ev else slice(None) for v in fam)
                        counts[node][sl] += wt * post.table
            delta = 0.0
            for node in range(self.n):
                new = self._normalise(counts[node] + self._pseudo(node, ess))
                delta = max(delta, float(np.abs(new - self.cpts[node]).max()))
                self.cpts[node] = new
            if verbose:
                print(f"  EM iter {it + 1}: max CPT change {delta:.5f}")
            if delta < tol:
                break
        return self

    @staticmethod
    def _normalise(counts):
        return counts / counts.sum(axis=-1, keepdims=True)

    # ---------------------------------------------------------------- inference
    def predict_proba(self, target: int, evidence: dict[int, int]) -> np.ndarray:
        return query(self, [target], evidence).table

    def log_likelihood_complete(self, X: np.ndarray) -> float:
        ll = 0.0
        for node in range(self.n):
            fam = self.parents[node] + [node]
            rows = X[(X[:, fam] >= 0).all(axis=1)][:, fam]
            ll += np.log(self.cpts[node][tuple(rows.T)]).sum()
        return float(ll)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        order = self._topological_order()
        X = np.zeros((n, self.n), dtype=np.int64)
        for node in order:
            ps = self.parents[node]
            for i in range(n):
                p = self.cpts[node][tuple(X[i, ps])]
                X[i, node] = rng.choice(self.card[node], p=p)
        return X

    def _topological_order(self):
        order, seen = [], set()
        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for p in self.parents[n]:
                visit(p)
            order.append(n)
        for n in range(self.n):
            visit(n)
        return order

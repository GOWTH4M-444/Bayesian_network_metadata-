"""Score-based structure learning (greedy hill climbing, BIC) tolerant to missing values."""
from __future__ import annotations
import numpy as np


def family_score(X: np.ndarray, card: list[int], child: int, parents: tuple[int, ...], penalty: float = 1.0) -> float:
    """BIC of one family on available cases, rescaled to the full sample size so scores of
    families with different amounts of missingness remain comparable."""
    N = X.shape[0]
    cols = [child] + list(parents)
    sub = X[:, cols]
    sub = sub[(sub >= 0).all(axis=1)]
    n = sub.shape[0]
    if n == 0:
        return 0.0
    r = card[child]
    q = int(np.prod([card[p] for p in parents])) if parents else 1
    idx = np.zeros(n, dtype=np.int64)
    for k, p in enumerate(parents):
        idx = idx * card[p] + sub[:, k + 1]
    counts = np.zeros((q, r))
    np.add.at(counts, (idx, sub[:, 0]), 1.0)
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(counts > 0, counts * np.log(counts / tot), 0.0).sum()
    bic = ll - penalty * 0.5 * np.log(n) * q * (r - 1)  # penalty<1 -> denser networks
    return float(bic * N / n)


def _has_path(parents: dict[int, set[int]], src: int, dst: int) -> bool:
    """True if a directed path src -> ... -> dst exists."""
    children: dict[int, list[int]] = {i: [] for i in parents}
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    stack, seen = [src], set()
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(children[n])
    return False


def learn_structure(X: np.ndarray, card: list[int], max_parents: int = 3, max_parent_configs: int = 2000,
                    whitelist: set[tuple[int, int]] | None = None,
                    blacklist: set[tuple[int, int]] | None = None,
                    max_iter: int = 500, verbose: bool = False, penalty: float = 1.0) -> dict[int, list[int]]:
    """Greedy hill climbing over add / delete / reverse edge moves.

    whitelist / blacklist hold (parent, child) index pairs: use them to inject domain knowledge,
    e.g. blacklist ("year" <- anything) or whitelist ("organism" -> "tissue")."""
    d = X.shape[1]
    whitelist, blacklist = whitelist or set(), blacklist or set()
    parents: dict[int, set[int]] = {i: set() for i in range(d)}
    cache: dict[tuple[int, frozenset], float] = {}

    def score(child, ps):
        key = (child, frozenset(ps))
        if key not in cache:
            cache[key] = family_score(X, card, child, tuple(sorted(ps)), penalty)
        return cache[key]

    def ok_size(ps):
        return len(ps) <= max_parents and np.prod([card[p] for p in ps] or [1]) <= max_parent_configs

    for (u, v) in whitelist:
        parents[v].add(u)

    for it in range(max_iter):
        best_gain, best_move = 1e-9, None
        for v in range(d):
            cur = score(v, parents[v])
            for u in range(d):
                if u == v:
                    continue
                if u in parents[v]:
                    if (u, v) not in whitelist:
                        gain = score(v, parents[v] - {u}) - cur  # delete
                        if gain > best_gain:
                            best_gain, best_move = gain, ("del", u, v)
                        # reverse u->v into v->u
                        if (v, u) not in blacklist and ok_size(parents[u] | {v}):
                            tmp = {k: set(s) for k, s in parents.items()}
                            tmp[v].discard(u)
                            if not _has_path(tmp, u, v):
                                gain = (score(v, parents[v] - {u}) - cur) + \
                                       (score(u, parents[u] | {v}) - score(u, parents[u]))
                                if gain > best_gain:
                                    best_gain, best_move = gain, ("rev", u, v)
                elif (u, v) not in blacklist and ok_size(parents[v] | {u}):
                    if not _has_path(parents, v, u):  # would create cycle
                        gain = score(v, parents[v] | {u}) - cur
                        if gain > best_gain:
                            best_gain, best_move = gain, ("add", u, v)
        if best_move is None:
            break
        op, u, v = best_move
        if op == "add":
            parents[v].add(u)
        elif op == "del":
            parents[v].discard(u)
        else:
            parents[v].discard(u)
            parents[u].add(v)
        if verbose:
            print(f"  iter {it + 1}: {op} {u}->{v}  (gain {best_gain:.1f})")
    return {k: sorted(s) for k, s in parents.items()}

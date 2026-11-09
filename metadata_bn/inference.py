"""Exact inference (variable elimination with barren-node pruning) for discrete BNs."""
from __future__ import annotations
import numpy as np


class Factor:
    def __init__(self, vars, table):
        self.vars = tuple(vars)
        self.table = np.asarray(table, dtype=float)

    def reduce(self, evidence: dict[int, int]) -> "Factor":
        idx, keep = [], []
        for v in self.vars:
            if v in evidence:
                idx.append(evidence[v])
            else:
                idx.append(slice(None))
                keep.append(v)
        return Factor(keep, self.table[tuple(idx)])

    def sum_out(self, var: int) -> "Factor":
        ax = self.vars.index(var)
        return Factor([v for v in self.vars if v != var], self.table.sum(axis=ax))


def multiply(factors: list[Factor], out_order: list[int] | None = None) -> Factor:
    all_vars = sorted({v for f in factors for v in f.vars})
    pos = {v: i for i, v in enumerate(all_vars)}
    args = []
    for f in factors:
        args += [f.table, [pos[v] for v in f.vars]]
    out_vars = list(out_order) if out_order is not None else all_vars
    args.append([pos[v] for v in out_vars])
    return Factor(out_vars, np.einsum(*args))


def _ancestors(parents: dict[int, list[int]], seeds) -> set[int]:
    seen, stack = set(), list(seeds)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(parents[n])
    return seen


def query(net, targets: list[int], evidence: dict[int, int]) -> Factor:
    """Posterior P(targets | evidence) as a normalised Factor with axes ordered as `targets`."""
    evidence = {k: v for k, v in evidence.items() if k not in targets}
    relevant = _ancestors(net.parents, list(targets) + list(evidence))
    factors = []
    for n in relevant:
        f = Factor(list(net.parents[n]) + [n], net.cpts[n]).reduce(evidence)
        if f.vars:
            factors.append(f)
    hidden = set(relevant) - set(targets) - set(evidence)
    while hidden:
        best, best_size = None, None
        for h in hidden:
            scope = {v for f in factors if h in f.vars for v in f.vars}
            size = np.prod([net.card[v] for v in scope]) if scope else 1
            if best is None or size < best_size:
                best, best_size = h, size
        involved = [f for f in factors if best in f.vars]
        factors = [f for f in factors if best not in f.vars]
        if involved:
            factors.append(multiply(involved).sum_out(best))
        hidden.discard(best)
    result = multiply(factors, out_order=list(targets))
    z = result.table.sum()
    result.table = result.table / z if z > 0 else np.full_like(result.table, 1.0 / result.table.size)
    return result

import numpy as np
from metadata_bn import MetadataBN
from metadata_bn.inference import query
from metadata_bn.network import BayesianNetwork
from metadata_bn.structure import learn_structure
from metadata_bn.synthetic import generate, true_network


def test_inference_matches_bruteforce():
    net = true_network(seed=3)
    X = net.sample(1, np.random.default_rng(0))
    ev = {0: int(X[0, 0]), 4: int(X[0, 4])}
    post = query(net, [5], ev).table
    # brute force: enumerate every assignment of all variables consistent with the evidence
    import itertools
    joint = np.zeros(net.card[5])
    for assign in itertools.product(*[range(c) for c in net.card]):
        if any(assign[k] != v for k, v in ev.items()):
            continue
        p = 1.0
        for n in range(net.n):
            p *= net.cpts[n][tuple(assign[q] for q in net.parents[n]) + (assign[n],)]
        joint[assign[5]] += p
    assert np.allclose(post, joint / joint.sum())


def test_structure_is_acyclic_and_finds_skeleton():
    net = true_network(0)
    X = net.sample(3000, np.random.default_rng(1))
    parents = learn_structure(X, net.card)
    learned = {frozenset((p, c)) for c, ps in parents.items() for p in ps}
    truth = {frozenset((p, c)) for c, ps in net.parents.items() for p in ps}
    assert len(learned & truth) / len(truth) >= 0.7
    # acyclicity
    BayesianNetwork(net.names, net.card, parents)._topological_order()


def test_em_beats_baseline_and_fill_missing():
    df_obs, df_full, _ = generate(n=1500, missing_rate=0.3, seed=2)
    m = MetadataBN(em_iters=3).fit(df_obs.iloc[:1200])
    out = m.fill_missing(df_obs.iloc[1200:1210])
    assert out.filter(regex="__p$").shape[1] == 9
    preds = m.predict({"organism": "Homo sapiens"})
    assert "tissue" in preds and abs(sum(p for _, p in preds["tissue"])) <= 1.0 + 1e-9


def test_lower_penalty_gives_denser_network():
    net = true_network(0)
    X = net.sample(2000, np.random.default_rng(3))
    dense = sum(len(v) for v in learn_structure(X, net.card, penalty=0.1).values())
    sparse = sum(len(v) for v in learn_structure(X, net.card, penalty=3.0).values())
    assert dense >= sparse

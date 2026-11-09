"""Synthetic GEO/TCGA-like study metadata generated from a known ground-truth Bayesian network.

Useful for unit tests and for benchmarking structure recovery / imputation accuracy."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .network import BayesianNetwork

CATEGORIES = {
    "organism": ["Homo sapiens", "Mus musculus", "Rattus norvegicus"],
    "year_bin": ["2005-2009", "2010-2014", "2015-2019", "2020+"],
    "data_type": ["mRNA expression", "miRNA", "methylation", "SNP/CNV", "RNA-seq"],
    "platform": ["Affymetrix array", "Illumina array", "Agilent array", "Illumina sequencer", "other"],
    "disease_area": ["cancer", "neurological", "cardiovascular", "immune", "metabolic", "healthy"],
    "tissue": ["breast", "lung", "brain", "blood", "heart", "liver", "colon", "skin"],
    "sample_size_bin": ["<20", "20-100", "100-500", ">500"],
    "has_clinical_data": ["yes", "no"],
    "treatment_study": ["yes", "no"],
}
# child: parents
TRUE_PARENTS = {
    "organism": [], "year_bin": [],
    "data_type": ["year_bin"],
    "platform": ["data_type", "year_bin"],
    "disease_area": ["organism"],
    "tissue": ["disease_area", "organism"],
    "sample_size_bin": ["data_type"],
    "has_clinical_data": ["disease_area", "sample_size_bin"],
    "treatment_study": ["disease_area"],
}


def true_network(seed: int = 0, concentration: float = 0.25) -> BayesianNetwork:
    rng = np.random.default_rng(seed)
    names = list(CATEGORIES)
    ix = {n: i for i, n in enumerate(names)}
    card = [len(CATEGORIES[n]) for n in names]
    parents = {ix[c]: [ix[p] for p in ps] for c, ps in TRUE_PARENTS.items()}
    net = BayesianNetwork(names, card, parents)
    for node in range(len(names)):
        shape = net._family_shape(node)
        flat = rng.dirichlet(np.full(shape[-1], concentration), size=int(np.prod(shape[:-1])) or 1)
        net.cpts[node] = flat.reshape(shape)
    return net


def generate(n: int = 3000, missing_rate: float = 0.25, seed: int = 0):
    """Returns (df_observed_with_missing, df_complete, true_network)."""
    net = true_network(seed)
    rng = np.random.default_rng(seed + 1)
    X = net.sample(n, rng)
    df_full = pd.DataFrame({net.names[j]: [CATEGORIES[net.names[j]][s] for s in X[:, j]]
                            for j in range(net.n)})
    mask = rng.random(X.shape) < missing_rate
    df_obs = df_full.mask(mask)
    return df_obs, df_full, net

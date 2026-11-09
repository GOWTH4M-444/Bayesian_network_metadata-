<div align="center">

<!-- Wave Banner Top -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=metadata-bn&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=36&desc=Bayesian%20networks%20for%20biomedical%20metadata%20completion&descSize=16&descAlignY=58&descColor=cdd6f4" width="100%"/>

<!-- Animated Typing -->
<a href="https://github.com/YOUR-USERNAME/metadata-bn">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=16&duration=3000&pause=1000&color=7C3AED&center=true&vCenter=true&multiline=true&width=700&height=60&lines=Learn+how+study+annotations+depend+on+each+other...;Then+fill+in+the+blanks%2C+with+probabilities." alt="Typing SVG"/>
</a>

<br/>

<!-- Dynamic Badges Row 1 — repo vitals -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/YOUR-USERNAME/metadata-bn?style=for-the-badge&color=22c55e)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/YOUR-USERNAME/metadata-bn/tests.yml?style=for-the-badge&label=Tests&logo=github-actions&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn/actions)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-f97316?style=for-the-badge)](https://github.com/YOUR-USERNAME/metadata-bn)

<!-- Dynamic Badges Row 2 — live repo metrics -->
[![Stars](https://img.shields.io/github/stars/YOUR-USERNAME/metadata-bn?style=for-the-badge&logo=starship&color=fbbf24&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn/stargazers)
[![Forks](https://img.shields.io/github/forks/YOUR-USERNAME/metadata-bn?style=for-the-badge&logo=git&color=818cf8&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn/network/members)
[![Issues](https://img.shields.io/github/issues/YOUR-USERNAME/metadata-bn?style=for-the-badge&logo=github&color=f43f5e&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn/issues)
[![Last Commit](https://img.shields.io/github/last-commit/YOUR-USERNAME/metadata-bn?style=for-the-badge&logo=git&color=34d399&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn/commits)
[![Repo Size](https://img.shields.io/github/repo-size/YOUR-USERNAME/metadata-bn?style=for-the-badge&logo=files&color=60a5fa&logoColor=white)](https://github.com/YOUR-USERNAME/metadata-bn)

</div>

---

## 📌 Table of Contents

<div align="center">

| | Section | | Section |
|:-:|:--------|:-:|:--------|
| 🧬 | [Overview](#-overview) | 🚀 | [Quick Start](#-quick-start) |
| 🔬 | [How It Works](#-how-it-works) | 🧪 | [Library Usage](#-library-usage) |
| 📊 | [Results](#-results) | 🗂 | [Project Structure](#-project-structure) |
| 💾 | [Data](#-data) | ⚙️ | [Method Details](#%EF%B8%8F-method-details) |
| ⚠️ | [Limitations](#%EF%B8%8F-limitations) | 🗺 | [Roadmap](#-roadmap) |
| 🤝 | [Contributing](#-contributing) | 📄 | [License](#-license) |

</div>

---

## 🧬 Overview

<table>
<tr>
<td>

Public repositories such as **[GEO](https://www.ncbi.nlm.nih.gov/geo/)** hold hundreds of thousands of molecular studies — but many are poorly annotated. Tissue, disease, or technology fields are frequently missing, making data hard to discover and reuse.

Metadata fields are **not independent**. A human cancer study tends to use particular platforms, sample sizes, and tissues. **metadata-bn** learns those dependencies as a Bayesian network and uses them to predict what is missing.

</td>
<td align="center" width="300">

```
 Known fields
 ┌──────────────┐
 │ organism: H. │
 │ type: expr.  │  →  🧠 Bayesian
 │ year: 2021   │      Network
 │ tissue: ???  │
 │ disease: ??? │  →  Predictions
 └──────────────┘      + confidence
```

</td>
</tr>
</table>

### 🛠 Tech Stack

<div align="center">

[![My Skills](https://skillicons.dev/icons?i=python,sklearn,github,git&theme=dark)](https://skillicons.dev)

| Library | Role |
|---------|------|
| `pgmpy` | Bayesian network structure + parameters |
| `scikit-learn` | Random forest baseline, preprocessing |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` / `networkx` | Network visualisation |
| `sqlite3` | GEOmetadb access |

</div>

---

## 🔬 How It Works

```mermaid
flowchart LR
    A[(GEOmetadb\n.sqlite)] -->|extract_geo.py| B[Raw fields\nper GEO series]
    B -->|Encode| C[Integer\ncategories\nmissing = -1]
    C -->|Hill-climbing\n+ BIC| D[Graph\nStructure]
    D -->|EM on\nincomplete rows| E[Conditional\nProbability\nTables]
    E -->|Variable\nelimination| F[Exact\nInference]
    F --> G([Predictions\n+ Confidence])

    style A fill:#1e1b4b,color:#c4b5fd,stroke:#7c3aed
    style G fill:#14532d,color:#86efac,stroke:#22c55e
    style D fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style E fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style F fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
```

<details>
<summary><b>🔍 Step-by-step breakdown</b></summary>

<br/>

| # | Step | What it does |
|---|------|--------------|
| 1 | **Extract** | Pulls one row per GEO series: organism, platform, study type, year, sample size, species/cell-line/perturbation flags, tissue and disease from sample annotations. |
| 2 | **Encode** | Categorical fields → integers; `missing` / `unknown` / `NA` → `−1`; rare labels merged into `__other__`. |
| 3 | **Learn structure** | Greedy hill-climbing on BIC, tolerant of missing values; whitelist/blacklist edges for domain knowledge; density auto-tuned on a validation split. |
| 4 | **Fit parameters** | Dirichlet-smoothed CPTs, then EM over unique row patterns so partially-annotated studies still contribute. |
| 5 | **Predict** | Exact variable elimination with barren-node pruning → a full probability distribution for every missing field. |
| 6 | **Explain** | Bootstrap edge stability, Markov blankets, and an annotated network diagram. |

</details>

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/metadata-bn.git
cd metadata-bn

# 2. Install
pip install -r requirements.txt

# 3. Run synthetic demo (no data needed — works in seconds)
python demo.py

# 4. Run unit tests
python -m pytest -q
```

> [!TIP]
> No data download needed for the demo — `demo.py` generates synthetic studies internally.

**On real GEO data** *(requires [`GEOmetadb.sqlite`](#-data), ~19 GB)*:

```bash
python run_all.py --quick      # ~1 min test on 3 000 studies
python run_all.py --n 60000    # full run: tuning + RF comparison + network diagram
```

<details>
<summary><b>📁 Output files</b></summary>

<br/>

| File | Description |
|------|-------------|
| `results/report.txt` | Accuracy summary and structure statistics |
| `results/network.png` | Learned Bayesian network diagram |
| `results/imputed.csv` | Full dataset with imputed values + confidence columns |
| `results/model.pkl` | Serialized trained model (load with `pickle`) |

</details>

---

## 🧪 Library Usage

```python
from metadata_bn import MetadataBN
import pandas as pd

# ── Fit ────────────────────────────────────────────────────
df    = pd.read_csv("studies.csv")   # one row per study
model = MetadataBN().fit(df)

# ── Predict missing fields ─────────────────────────────────
preds = model.predict({
    "organism":     "Homo sapiens",
    "is_cell_line": "no",
    "disease_area": "cancer",
})
# → {"tissue": [("liver", 0.31), ("lung", 0.18), ...], ...}

# ── Impute an entire dataframe ─────────────────────────────
df_filled = model.fill_missing(df)
# Adds imputed values + per-cell confidence columns

# ── Inspect the learned dependencies ──────────────────────
model.markov_blanket("tissue")
# → ["organism", "disease_area", "platform_technology", ...]

# ── Save & reload ─────────────────────────────────────────
model.save("results/model.pkl")
model = MetadataBN.load("results/model.pkl")
```

---

## 📊 Results

> [!NOTE]
> Results below are from a **3,000-study quick run** (~200–600 test cells per field). Differences of a few points are within noise. Treat as preliminary.

Each observed value in a held-out study is hidden and predicted from the study's remaining fields.

### Accuracy vs. Baselines

| Field | Majority class | Bayesian Network | Random Forest |
|-------|:-:|:-:|:-:|
| organism | `0.41` | `0.76` ▲ | `0.84` |
| platform technology | `0.45` | `0.81` ▲ | `0.92` |
| study type | `0.41` | `0.61` ▲ | `0.74` |
| tissue | `0.17` | `0.25` ▲ | `0.27` |
| disease area | `0.57` | `0.63` ▲ | `0.63` |

**Bayesian network accuracy (visual)**

```
organism            ████████████████████████████████████████▌  0.76
platform tech       █████████████████████████████████████████████  0.81
study type          ██████████████████████████████▌  0.61
tissue              ████████████▌  0.25
disease area        ██████████████████████████████████▌  0.63
                    0    0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9
```

### Why use a Bayesian network over a random forest?

<div align="center">

| Capability | Random Forest | Bayesian Network |
|-----------|:---:|:---:|
| Top-1 accuracy | ✅ Higher | ✅ Competitive |
| Native missing-value handling | ❌ | ✅ |
| Full probability distribution | ❌ | ✅ |
| Interpretable dependencies | ❌ | ✅ |
| Queryable with partial evidence | ❌ | ✅ |
| Lightweight inference | ❌ | ✅ |

</div>

> The network's real value is **interpretability + uncertainty quantification**, not just top-1 accuracy. Run `python demo.py` to see the learned network diagram.

---

## 🗂 Project Structure

```
metadata-bn/
│
├── 📦 metadata_bn/              # Core library
│   ├── data.py                  #   Category encoding, missing values, rare-label merging
│   ├── structure.py             #   Hill-climbing structure learning (BIC, white/blacklists)
│   ├── network.py               #   CPTs, EM for incomplete data, sampling
│   ├── inference.py             #   Exact variable elimination
│   ├── model.py                 #   High-level API — fit / predict / fill_missing
│   ├── tune.py                  #   Automatic network-density selection
│   ├── evaluate.py              #   Held-out imputation accuracy, structure recovery
│   ├── benchmark.py             #   Random-forest comparison
│   ├── viz.py                   #   Bootstrap edge confidence + network diagram
│   ├── vocab.py                 #   Tissue / disease / species synonym vocabularies
│   └── synthetic.py             #   Ground-truth generator for tests and demos
│
├── 🔧 extract_geo.py            # GEOmetadb.sqlite → studies.csv
├── 🔧 run_pipeline.py           # Train + evaluate + impute on any CSV
├── 🔧 run_all.py                # End-to-end: extract → train → evaluate → report
├── 🔧 demo.py                   # Synthetic end-to-end demo (no data needed)
├── 🔧 predict_example.py        # Query a trained model
│
└── 🧪 tests/                    # Unit and integration tests
```

---

## 💾 Data

> [!IMPORTANT]
> `GEOmetadb.sqlite` (~19 GB) is **not** included. You must download it separately.

**Download via R:**

```r
install.packages("BiocManager")
BiocManager::install("GEOmetadb")
library(GEOmetadb)
getSQLiteFile()   # → saves GEOmetadb.sqlite in working directory
```

Place the file in the project root, or pass `--db /path/to/GEOmetadb.sqlite` to any script. `run_all.py` also searches your Downloads folder automatically.

**Using your own data:**

```bash
# Any CSV of categorical study metadata works out of the box
python run_pipeline.py my_studies.csv
```

---

## ⚙️ Method Details

<details>
<summary><b>Encoding</b></summary>

Categorical fields are mapped to integers. Missing values, `"unknown"`, and `"NA"` are encoded as `−1`. Labels with fewer than `min_count` occurrences are merged into an `__other__` category to prevent rare-label fragmentation.

</details>

<details>
<summary><b>Structure Learning</b></summary>

Greedy search over add / delete / reverse edge moves with acyclicity enforced at each step. Family BIC is computed on available cases and rescaled to the full sample size, keeping fields with different missingness rates comparable. A `penalty` factor (< 1 gives denser networks) and `max_parents` are selected by validation accuracy when `--tune` is enabled.

</details>

<details>
<summary><b>Parameter Estimation</b></summary>

BDeu-style Dirichlet smoothing is applied to conditional probability tables. EM is then run over unique row patterns, assuming data is **missing at random (MAR)** — this allows partially-annotated studies to contribute to every parameter.

</details>

<details>
<summary><b>Inference</b></summary>

Exact variable elimination with barren-node pruning and a minimum-size elimination order. Correctness is verified against brute-force enumeration in the test suite.

</details>

<details>
<summary><b>Evaluation Protocol</b></summary>

Every observed cell of held-out studies is hidden in turn and predicted from the study's remaining fields. Predictions are benchmarked against a majority-class baseline and a scikit-learn random forest (100 trees, one model per target field).

</details>

<details>
<summary><b>Edge Confidence (Bootstrap)</b></summary>

Network structure is re-learned on bootstrap resamples of the training data. The stability score for each directed edge is the fraction of resamples in which that exact edge (including direction) reappears. Reported in `results/report.txt` and visualised as arrow width in `results/network.png`.

</details>

---

## ⚠️ Limitations

> [!CAUTION]
> Please read before interpreting any results.

| Limitation | Detail |
|------------|--------|
| **Noisy labels** | Tissue and disease labels are from synonym matching on free text (`vocab.py`), not curated ontologies. Evaluation measures agreement with these noisy labels. |
| **MAR assumption** | Real annotators skip fields systematically, violating the missing-at-random assumption used in EM. |
| **Categorical only** | Numeric fields must be binned first. Fields with hundreds of categories (e.g. raw platform ID) perform poorly. |
| **Non-causal edges** | Edge directions are not reliably causal — observational data cannot always orient edges. |
| **GEO only** | Developed and tested on GEO; generalization to other repositories is untested. |

---

## 🗺 Roadmap

```mermaid
flowchart LR
    A([✅ Structure learning\nEM · exact inference\nunit tests]) --> B([✅ GEOmetadb extraction\nsample-level annotations])
    B --> C([✅ RF comparison\nedge stability\ndiagram · auto-tune])
    C --> D([🔲 Ontology mapping\nUBERON · MONDO])
    D --> E([🔲 Text features\ntitle/abstract\nLLM topic tags])
    E --> F([🔲 Molecular features\npredicted sex\ntissue signatures])
    F --> G([🔲 MNAR modelling])
    G --> H([🔲 Multi-repo support\nTCGA · ArrayExpress])

    style A fill:#14532d,color:#86efac,stroke:#22c55e
    style B fill:#14532d,color:#86efac,stroke:#22c55e
    style C fill:#14532d,color:#86efac,stroke:#22c55e
    style D fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style E fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style F fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style G fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
    style H fill:#1e3a5f,color:#93c5fd,stroke:#3b82f6
```

<!-- Wave Banner Bottom -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>
  <sub>Built for open science · Contributions and feedback welcome · <a href="https://github.com/YOUR-USERNAME/metadata-bn/issues">Open an issue</a></sub>
</div>

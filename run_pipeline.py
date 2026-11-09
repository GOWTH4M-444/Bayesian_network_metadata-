"""Train + evaluate on a metadata CSV (e.g. the output of extract_geo.py) in one command.

  python run_pipeline.py studies.csv --columns organism platform_tech ... tissue disease_area

Writes model.pkl, imputed.csv, network.png and report.txt (in --out-dir).
Needs scikit-learn + matplotlib for the random-forest comparison and the diagram (pip install scikit-learn matplotlib)."""
import argparse
import pandas as pd
from metadata_bn import MetadataBN
from metadata_bn.evaluate import evaluate_observed

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--columns", nargs="*")
ap.add_argument("--min-count", type=int, default=20)
ap.add_argument("--max-parents", type=int, default=3)
ap.add_argument("--test-frac", type=float, default=0.2)
ap.add_argument("--em-iters", type=int, default=5)
ap.add_argument("--tune", action="store_true", help="auto-select network density on a validation split")
ap.add_argument("--n-boot", type=int, default=15, help="bootstrap resamples for edge confidence")
ap.add_argument("--no-extras", action="store_true", help="skip random forest comparison, bootstrap and diagram")
ap.add_argument("--out-dir", default=".")
a = ap.parse_args()

pd.set_option("display.width", 160)
df = pd.read_csv(a.csv, dtype=str)
cols = a.columns or [c for c in df.columns if c.lower() not in ("gse", "id")]
df = df[cols].sample(frac=1.0, random_state=0).reset_index(drop=True)
n_test = int(len(df) * a.test_frac)
test, train = df.iloc[:n_test], df.iloc[n_test:]
print(f"{len(train)} training studies, {len(test)} held-out studies")

penalty, max_parents, tune_table = 1.0, a.max_parents, None
if a.tune:
    from metadata_bn.tune import auto_tune
    print("Tuning network density on a validation split ...")
    penalty, max_parents, tune_table = auto_tune(train, cols, a.min_count)
    print(tune_table.round(3).to_string(index=False))
    print(f"-> using penalty={penalty}, max_parents={max_parents}")
model = MetadataBN(min_count=a.min_count, max_parents=max_parents, penalty=penalty,
                   em_iters=a.em_iters, verbose=True).fit(train, cols)
res = evaluate_observed(model, test, train)
conf = None

if not a.no_extras:
    try:
        from metadata_bn.benchmark import evaluate_random_forest
        print("\nTraining random-forest comparison ...")
        res = res.join(evaluate_random_forest(model, test, train))
    except ImportError:
        print("(skipping random forest: run  pip install scikit-learn )")
    try:
        from metadata_bn.viz import bootstrap_edges, edge_confidence_table, plot_network
        print(f"Bootstrapping edge confidence ({a.n_boot} resamples) ...")
        boot = bootstrap_edges(train, cols, a.min_count, max_parents, n_boot=a.n_boot, penalty=penalty)
        conf = edge_confidence_table(model, boot)
        plot_network(model, boot, f"{a.out_dir}/network.png")
    except ImportError:
        print("(skipping diagram: run  pip install matplotlib )")

res["BN gain vs baseline"] = res["BN top-1"] - res["baseline (majority)"]
lines = (["NETWORK DENSITY TUNING (validation split):", tune_table.round(3).to_string(index=False), ""] if tune_table is not None else []) + ["LEARNED EDGES and how stable they are across resampled data:",
         (conf.round(2).to_string() if conf is not None else "\n".join(f"  {p} -> {c}" for p, c in model.edges())),
         "", "HELD-OUT ACCURACY (each observed value hidden, predicted from the study's other fields):",
         res.round(3).to_string(),
         "", "Read it as: BN = Bayesian network; baseline = always guess the most common value;",
         "gain = BN top-1 minus baseline. Gains near 0 mean those fields carry no information about each other."]
report = "\n".join(lines)
print("\n" + report)
open(f"{a.out_dir}/report.txt", "w").write(report)
model.save(f"{a.out_dir}/model.pkl")
print("\nImputing missing cells for the full file ...")
model.fill_missing(df).to_csv(f"{a.out_dir}/imputed.csv", index=False)
print("Done: model.pkl, imputed.csv, report.txt" + ("" if a.no_extras else ", network.png"))

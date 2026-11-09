"""End-to-end demo on synthetic GEO-like metadata with 25% missing values."""
import time
import pandas as pd
from metadata_bn import MetadataBN
from metadata_bn.synthetic import generate, TRUE_PARENTS
from metadata_bn.evaluate import imputation_accuracy, structure_recovery

pd.set_option("display.width", 140)

df_obs, df_full, truth = generate(n=3600, missing_rate=0.25, seed=0)
train_obs, test_full = df_obs.iloc[:3000], df_full.iloc[3000:]

t = time.time()
model = MetadataBN(min_count=5, max_parents=3, em_iters=8, verbose=True).fit(train_obs)
print(f"\nTrained in {time.time() - t:.1f}s")

print("\nLearned edges:")
for a, b in model.edges():
    print(f"  {a} -> {b}")

truth_edges = {(p, c) for c, ps in TRUE_PARENTS.items() for p in ps}
print("\nStructure recovery:", structure_recovery(set(model.edges()), truth_edges))

print("\nHeld-out imputation (each field hidden in turn, predicted from all others):")
print(imputation_accuracy(model, test_full, train_obs).round(3))

print("\nExample: partially annotated study")
record = {"organism": "Homo sapiens", "data_type": "methylation", "disease_area": "cancer"}
print(" given:", record)
for field, top in model.predict(record, top_k=3).items():
    print(f"  {field:18s}", ", ".join(f"{v} ({p:.2f})" for v, p in top))

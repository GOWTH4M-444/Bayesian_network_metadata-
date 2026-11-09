"""ONE COMMAND:  python run_all.py

Finds GEOmetadb.sqlite, extracts studies (with sample-level tissue/disease), trains the Bayesian
network, evaluates it on held-out studies, fills in missing metadata, and writes everything to ./results

Options:  --db PATH   --n 30000 (number of studies)   --quick (small & fast test run)   --no-sample-level
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import extract_geo

COLUMNS = ["organism", "platform_tech", "platform_vendor", "study_type", "year", "sample_size_bin",
           "species_in_text", "is_cell_line", "has_perturbation", "is_time_series", "tissue", "disease_area"]


def find_db():
    found = []
    home_dl = Path.home() / "Downloads"
    for base, recursive in [(Path.cwd(), False), (Path.cwd().parent, False), (home_dl, True)]:
        if base.exists():
            for f in (base.rglob("GEOmetadb*.sqlite") if recursive else base.glob("GEOmetadb*.sqlite")):
                if f.is_file() and f.stat().st_size >= extract_geo.MIN_BYTES:
                    found.append(f)
    return max(found, key=lambda f: f.stat().st_size) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-sample-level", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    db = Path(a.db) if a.db else find_db()
    if not db or not db.is_file():
        sys.exit("Could not find GEOmetadb.sqlite. Run:  python run_all.py --db \"C:\\path\\to\\GEOmetadb.sqlite\"")
    n = 3000 if a.quick else a.n
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[1/3] Extracting {n} studies from {db}")
    extract_geo.main(str(db), str(out / "studies.csv"), limit=n, sample_level=not a.no_sample_level)
    print(f"      done ({(time.time() - t0) / 60:.1f} min)\n[2/3] Training + evaluating + imputing")
    import pandas as pd
    have = pd.read_csv(out / "studies.csv", nrows=1).columns
    cols = [c for c in COLUMNS if c in have]
    r = subprocess.run([sys.executable, "run_pipeline.py", str(out / "studies.csv"), "--columns", *cols,
                        "--out-dir", str(out), "--em-iters", "3" if a.quick else "5"]
                       + ([] if (a.quick or a.no_tune) else ["--tune"]))
    if r.returncode:
        sys.exit("Training step failed - see the error above.")
    print(f"\n[3/3] ALL DONE in {(time.time() - t0) / 60:.1f} min. Results are in the folder: {out.resolve()}")
    print("  report.txt   - learned network + accuracy vs. baseline")
    print("  imputed.csv  - every study with missing metadata filled in (+ confidence columns '__p')")
    print("  model.pkl    - trained model (use predict_example.py to query it)")


if __name__ == "__main__":
    main()

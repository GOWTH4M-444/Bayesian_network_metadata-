"""Command line:  python -m metadata_bn.cli train studies.csv model.pkl
                 python -m metadata_bn.cli impute model.pkl studies.csv filled.csv"""
import argparse
import pandas as pd
from .model import MetadataBN


def main():
    ap = argparse.ArgumentParser(prog="metadata_bn")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("csv"); t.add_argument("out")
    t.add_argument("--columns", nargs="*", help="metadata columns to model (default: all)")
    t.add_argument("--max-parents", type=int, default=3)
    t.add_argument("--min-count", type=int, default=5)
    i = sub.add_parser("impute")
    i.add_argument("model"); i.add_argument("csv"); i.add_argument("out")
    i.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "train":
        df = pd.read_csv(a.csv)
        m = MetadataBN(min_count=a.min_count, max_parents=a.max_parents, verbose=True).fit(df, a.columns)
        m.save(a.out)
        print("Edges:", *[f"{p}->{c}" for p, c in m.edges()], sep="\n  ")
    else:
        m = MetadataBN.load(a.model)
        m.fill_missing(pd.read_csv(a.csv), a.min_confidence).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

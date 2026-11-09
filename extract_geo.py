"""Extract one row per GEO series (GSE) from GEOmetadb.sqlite into studies.csv.

Usage:  python extract_geo.py path/to/GEOmetadb.sqlite studies.csv [--limit 50000] [--sample-level]
  --sample-level  also read per-sample source/characteristics text (better tissue/disease; slower)

Columns produced (all categorical; unknown -> empty = missing):
  organism, platform, platform_tech, platform_vendor, study_type, year, sample_size_bin,
  species_in_text, is_cell_line, has_perturbation, is_time_series, tissue, disease_area
tissue / disease_area come from synonym matching (see metadata_bn/vocab.py): sample text first, then the study
text, then cell-line -> tissue of origin. Edit vocab.py to improve the labels."""
import os
import re
import sqlite3
import sys
import pandas as pd

MIN_BYTES = int(os.environ.get("GEO_MIN_BYTES", 100_000_000))
from metadata_bn.vocab import (TISSUE, CELL_LINE_TISSUE, DISEASE, SPECIES_TEXT, CELL_LINE, TREATMENT,
                              TIMECOURSE, flag, first_match)

def size_bin(n):
    if n is None or pd.isna(n): return None
    return "<20" if n < 20 else "20-100" if n < 100 else "100-500" if n < 500 else ">500"

def ensure_index(con, table, col):
    """Add an index on table(col) if none starts with it (makes the big joins fast). Safe to repeat."""
    for idx in con.execute(f"pragma index_list({table})").fetchall():
        cols = [r[2] for r in con.execute(f"pragma index_info({idx[1]})").fetchall()]
        if cols and cols[0] == col:
            return
    print(f"  building index on {table}({col}) - one-off, may take a few minutes ...")
    con.execute(f"create index if not exists idx_bn_{table}_{col} on {table}({col})")
    con.commit()


def sample_info(con, gse_ids, per_series=5):
    """Per study: text (source name + characteristics of up to `per_series` samples), plus the platform
    and organism recorded on its samples (used to fill gaps in the study-level platform links)."""
    ensure_index(con, "gse_gsm", "gse")
    ensure_index(con, "gsm", "gsm")
    cols = {r[1] for r in con.execute("pragma table_info(gsm)")}
    q_gpl = "gsm.gpl as gpl_s" if "gpl" in cols else "null as gpl_s"
    q_org = "gsm.organism_ch1 as org_s" if "organism_ch1" in cols else "null as org_s"
    con.execute("create temp table sel(gse text primary key)")
    con.executemany("insert or ignore into sel values (?)", [(g,) for g in gse_ids])
    d = pd.read_sql_query(
        f"select sel.gse as gse, gsm.source_name_ch1 as src, gsm.characteristics_ch1 as ch, {q_gpl}, {q_org} "
        "from sel join gse_gsm on gse_gsm.gse = sel.gse join gsm on gsm.gsm = gse_gsm.gsm", con)
    d = d.groupby("gse").head(per_series).copy()
    d["t"] = d.src.fillna("") + " " + d.ch.fillna("").str.replace(r";\s*", " ", regex=True)

    def mode(s):
        s = s.dropna()
        s = s[s.astype(str).str.strip() != ""]
        return s.mode().iloc[0] if len(s) else None

    g = d.groupby("gse")
    return pd.DataFrame({"text": g.t.apply(" ".join).str.lower(),
                         "gpl_s": g.gpl_s.apply(mode), "org_s": g.org_s.apply(mode)})


def main(db, out, limit=None, sample_level=False):
    if not os.path.isfile(db):
        sys.exit(f"File not found: {db}\nCheck the path (run: dir \"folder\\GEOmetadb*\" to see the real filename).")
    if os.path.getsize(db) < MIN_BYTES:
        sys.exit(f"{db} is only {os.path.getsize(db)/1e6:.1f} MB - the download looks incomplete or it is a .gz file that needs unzipping.")
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    for t in ("gse", "gpl", "gse_gpl", "gse_gsm"):
        if t not in tables:
            sys.exit(f"Table {t!r} not found. Tables present: {sorted(tables)}")
    q = f"""
      SELECT gse.gse, gse.title, gse.summary, gse.overall_design, gse.type, gse.submission_date
      FROM gse {'ORDER BY RANDOM() LIMIT ' + str(int(limit)) if limit else ''}"""
    df = pd.read_sql_query(q, con)
    print(f"{len(df)} series read")
    gpl = pd.read_sql_query(
        "SELECT gse_gpl.gse, gse_gpl.gpl AS platform, gpl.organism, gpl.manufacturer, gpl.technology FROM gse_gpl JOIN gpl ON gpl.gpl = gse_gpl.gpl", con
    ).drop_duplicates("gse")
    n = pd.read_sql_query("SELECT gse, COUNT(*) AS n FROM gse_gsm GROUP BY gse", con)
    df = df.merge(gpl, on="gse", how="left").merge(n, on="gse", how="left")

    stext = pd.Series("", index=df.index)
    if sample_level:
        print("Reading sample-level annotations (slow on the full database) ...")
        info = sample_info(con, df.gse.tolist())
        stext = df.gse.map(info.text).fillna("")
        # fill platform / organism gaps from the samples' own records
        before = df.platform.isna().mean()
        tab = pd.read_sql_query("select gpl, organism, manufacturer, technology from gpl", con).drop_duplicates("gpl").set_index("gpl")
        miss = df.platform.isna()
        df.loc[miss, "platform"] = df.loc[miss, "gse"].map(info.gpl_s)
        for c in ("organism", "manufacturer", "technology"):
            df[c] = df[c].fillna(df.platform.map(tab[c]))
        df["organism"] = df["organism"].fillna(df.gse.map(info.org_s))
        print(f"  platform missing: {before:.0%} -> {df.platform.isna().mean():.0%} after using sample records")
    text = (df.title.fillna("") + " " + df.summary.fillna("") + " " + df.overall_design.fillna("")).str.lower()
    both = stext + " " + text
    res = pd.DataFrame({
        "gse": df.gse,
        "organism": df.organism.str.split(";").str[0].str.strip(),
        "platform": df.platform,
        "platform_tech": df.technology.str.lower().str.strip(),
        "platform_vendor": df.manufacturer.str.split().str[0].str.lower().str.strip(",."),
        "study_type": df.type.str.split(";").str[0].str.strip(),
        "year": pd.to_datetime(df.submission_date, errors="coerce").dt.year.astype("Int64").astype(str).replace("<NA>", None),
        "sample_size_bin": df.n.map(size_bin),
        "species_in_text": both.map(lambda t: first_match(t, SPECIES_TEXT)),
        "is_cell_line": both.map(lambda t: flag(t, CELL_LINE)),
        "has_perturbation": text.map(lambda t: flag(t, TREATMENT)),
        "is_time_series": text.map(lambda t: flag(t, TIMECOURSE)),
        "tissue": [first_match(a, TISSUE) or first_match(b, TISSUE) or first_match(c, CELL_LINE_TISSUE)
                   for a, b, c in zip(stext, text, both)],
        "disease_area": [first_match(a, DISEASE) or first_match(b, DISEASE) for a, b in zip(stext, text)],
    })
    res.to_csv(out, index=False)
    print(f"Wrote {out}\n\nFraction missing per column:")
    print(res.drop(columns="gse").isna().mean().round(2).to_string())

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    if lim is not None: args = [a for a in args if a != str(lim)]
    main(args[0], args[1], lim, sample_level="--sample-level" in sys.argv)

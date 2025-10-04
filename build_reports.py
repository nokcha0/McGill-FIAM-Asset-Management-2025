"""
build_reports.py
Aggregate raw SEC filings / text data into a single reports dataset.
Steps:
  1. Traverse yearly folders under TEXT_DIR
  2. Read each text_us_YYYY.pkl file
  3. Concatenate into a single dataframe
  4. Save as reports.parquet for nlp_engine.py
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "/teamspace/uploads")
TEXT_DIR     = os.environ.get("TEXT_DIR", "/teamspace/text_data")   # folder containing year subfolders
OUT_FILE     = os.path.join("output", "data", "reports.parquet")

os.makedirs("output/data", exist_ok=True)

all_dfs = []
years = sorted([d for d in os.listdir(TEXT_DIR) if d.isdigit()])

for y in years:
    year_path = os.path.join(TEXT_DIR, y)
    if not os.path.isdir(year_path):
        continue
    for fname in os.listdir(year_path):
        if fname.endswith(".pkl") and "text_us" in fname.lower():
            fpath = os.path.join(year_path, fname)
            print(f"Loading {fpath}")
            try:
                df = pd.read_pickle(fpath)
                # normalize column names
                df.columns = [c.lower() for c in df.columns]
                keep = {}
                if "gvkey" in df.columns: keep["id"] = df["gvkey"].astype(str)
                elif "id" in df.columns: keep["id"] = df["id"].astype(str)
                if "datadate" in df.columns: keep["char_eom"] = pd.to_datetime(df["datadate"], errors="coerce")
                elif "char_eom" in df.columns: keep["char_eom"] = pd.to_datetime(df["char_eom"], errors="coerce")
                if "text" in df.columns: keep["text"] = df["text"].astype(str)
                elif "content" in df.columns: keep["text"] = df["content"].astype(str)
                if keep:
                    all_dfs.append(pd.DataFrame(keep))
            except Exception as e:
                print(f"    -> Failed to load {fpath}: {e}")

if not all_dfs:
    raise RuntimeError("No reports loaded from TEXT_DIR")

reports = pd.concat(all_dfs, ignore_index=True).dropna(subset=["id","char_eom","text"])
reports = reports.drop_duplicates(subset=["id","char_eom","text"])

# save as parquet
table = pa.Table.from_pandas(reports, preserve_index=False)
pq.write_table(table, OUT_FILE, compression="snappy")

print(f"[DONE] Wrote {OUT_FILE} with shape {reports.shape}")

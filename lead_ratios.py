"""
This code snippet takes a 4-month lead of the accounting ratios and adds back to the original sample
The lead ratios can be used as left-hand-side variables
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "/teamspace/uploads")

DATA_FILE   = os.path.join(TEMPLATE_DIR, "ret_sample.csv")
RATIOS_FILE = os.path.join(TEMPLATE_DIR, "acc_ratios.csv")
OUT_FILE    = "output/data/final_data.parquet"
RATIOS_TMP  = "output/cache/ratios_lead4.parquet"
CHUNK_SIZE  = 1_000_000

os.makedirs("output/data", exist_ok=True)
os.makedirs("output/cache", exist_ok=True)

# read the original sample header
header_cols = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
date_like   = [c for c in ["date","ret_eom","char_date","char_eom"] if c in header_cols]

# read the list of ratios
ratio_list = pd.read_csv(RATIOS_FILE)["Variable"].astype(str).tolist()

# keep the necessary identifiers
ratio_usecols = ["id","char_eom"] + [c for c in ratio_list if c in header_cols]

# move the ratios 4 month ahead
if os.path.exists(RATIOS_TMP):
    os.remove(RATIOS_TMP)

writer = None
for chunk in pd.read_csv(
    DATA_FILE, usecols=[c for c in ratio_usecols if c in header_cols],
    parse_dates=[c for c in ["char_eom"] if c in ratio_usecols],
    chunksize=CHUNK_SIZE, low_memory=False
):
    ratios = chunk.copy()
    ratios["char_eom"] = ratios["char_eom"] + pd.DateOffset(months=4)
    ratios = ratios.rename(columns={col: col + "_lead4" for col in ratio_list if col in ratios.columns})
    ratios = ratios.rename(columns={"char_eom": "ret_eom"})
    table = pa.Table.from_pandas(ratios, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(RATIOS_TMP, table.schema, compression="snappy")
    writer.write_table(table)
if writer is not None:
    writer.close()

# merge with the original data and save
base_needed = ["id","ret_eom","date","stock_ret","year","month"]
base_usecols = [c for c in base_needed if c in header_cols]
ratios_all = pd.read_parquet(RATIOS_TMP)

final_writer = None
for chunk in pd.read_csv(
    DATA_FILE, usecols=base_usecols,
    parse_dates=[c for c in ["date","ret_eom"] if c in base_usecols],
    chunksize=CHUNK_SIZE, low_memory=False
):
    final_data = pd.merge(chunk, ratios_all, on=["id","ret_eom"], how="left")
    table = pa.Table.from_pandas(final_data, preserve_index=False)
    if final_writer is None:
        final_writer = pq.ParquetWriter(OUT_FILE, table.schema, compression="snappy")
    final_writer.write_table(table)
if final_writer is not None:
    final_writer.close()

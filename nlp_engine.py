"""
nlp_engine.py
Extract advanced text-based financial factors and merge into baseline dataset.
Steps:
  1. Load baseline & reports
  2. Compute lexicon-based sentiment/risk features
  3. Generate novelty score using embeddings (GPU if available, with caching)
  4. Merge with baseline dataset
  5. Apply monthly z-scoring
"""

import os, re, hashlib
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import torch
import pyarrow.parquet as pq

BASELINE_FILE = "output/data/final_data.parquet"
REPORTS_FILE  = "output/data/reports.parquet"
OUT_FILE      = "output/data/final_data_text.parquet"
CACHE_DIR     = "output/cache"
EMB_CACHE     = os.path.join(CACHE_DIR, "embeddings.parquet")

POS = {"growth","improve","strength","robust","opportunity","profitable","innovation","expansion","efficiency","tailwind"}
NEG = {"decline","weak","risk","headwind","loss","impairment","litigation","regulatory","supply","inflation","recession"}
UNCERTAINTY = {"uncertain","volatility","approximately","could","may","might","pending","depends","risk of"}

RISK_REG   = {"regulation","regulatory","compliance","fine","penalty"}
RISK_LEGAL = {"litigation","lawsuit","settlement","patent dispute"}
RISK_SUPPLY= {"supply chain","supplier","shortage","logistics","freight"}
RISK_CYBER = {"cybersecurity","ransomware","breach"}
RISK_MACRO = {"inflation","rates","interest rate","currency","commodity"}
RISK_GEO   = {"geopolitical","sanction","tariff","war","conflict"}

COST_PRESS = {"cost pressure","wage","labor cost","input cost","freight cost"}
MARGIN_UP  = {"margin expansion","higher margin","improved margin"}
MARGIN_DN  = {"margin compression","lower margin","compressed margin"}
CAPEX_INV  = {"capex","capital expenditure","capacity expansion"}
AI_INV     = {"ai investment","machine learning","artificial intelligence"}
BUYBACK    = {"share repurchase","buyback"}
DIV_UP     = {"dividend increase","raise dividend"}
DIV_DN     = {"dividend cut","suspend dividend"}
LAYOFFS    = {"layoff","workforce reduction","severance"}
HIRING     = {"hiring","recruiting","headcount growth"}

NEGATIONS  = {"no","not","never","without"}
INTENS_UP  = {"significant","substantial","strong"}
INTENS_DN  = {"slight","modest","minor"}

TOKEN_RE = re.compile(r"[a-z0-9\-]+")
def tokenize(text): return TOKEN_RE.findall(str(text).lower())
def gen_bigrams(tokens): return [f"{a} {b}" for a, b in zip(tokens[:-1], tokens[1:])]

def sentiment_with_logic(tokens):
    pos, neg = 0.0, 0.0
    for i,t in enumerate(tokens):
        mult = 1.0
        if i>0 and tokens[i-1] in INTENS_UP: mult*=1.5
        if i>0 and tokens[i-1] in INTENS_DN: mult*=0.5
        hit_pos, hit_neg = t in POS, t in NEG
        if i>0 and tokens[i-1] in NEGATIONS: hit_pos, hit_neg = hit_neg, hit_pos
        if i+1<len(tokens) and tokens[i+1] in NEGATIONS: hit_pos, hit_neg = hit_neg, hit_pos
        if hit_pos: pos += mult
        if hit_neg: neg += mult
    return pos, neg

def per_k(val,length): return 1000.0*(val/max(length,1))
def hash_row(row):
    text = str(row.get("text","")) + str(row.get("id","")) + str(row.get("char_eom",""))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[1/5] Loading baseline + reports")
    schema_cols = set(pq.ParquetFile(BASELINE_FILE).schema.names)
    keep_cols = [c for c in ["id","ret_eom","date","stock_ret"] if c in schema_cols] + [c for c in schema_cols if c.endswith("_lead4")]
    base = pd.read_parquet(BASELINE_FILE, columns=keep_cols)

    reports = pd.read_parquet(REPORTS_FILE)
    reports["char_eom"] = pd.to_datetime(reports["char_eom"], errors="coerce")

    print("[2/5] Computing lexicon features")
    feats, hashes = [], []
    for _,row in reports.iterrows():
        toks = tokenize(row.get("text",""))
        bigr = gen_bigrams(toks)
        L = len(toks)
        sp,sn = sentiment_with_logic(toks)
        f = {
            "sent_pos_k": per_k(sp,L),
            "sent_neg_k": per_k(sn,L),
            "sent_net_k": per_k(sp-sn,L),
            "uncertainty_k": per_k(sum(t in UNCERTAINTY for t in toks),L),
            "risk_reg_k": per_k(sum(t in RISK_REG for t in toks+bigr),L),
            "risk_legal_k": per_k(sum(t in RISK_LEGAL for t in toks+bigr),L),
            "risk_supply_k": per_k(sum(t in RISK_SUPPLY for t in toks+bigr),L),
            "risk_cyber_k": per_k(sum(t in RISK_CYBER for t in toks+bigr),L),
            "risk_macro_k": per_k(sum(t in RISK_MACRO for t in toks+bigr),L),
            "risk_geo_k": per_k(sum(t in RISK_GEO for t in toks+bigr),L),
            "cost_pressure_k": per_k(sum(t in COST_PRESS for t in toks+bigr),L),
            "margin_up_k": per_k(sum(t in MARGIN_UP for t in toks+bigr),L),
            "margin_down_k": per_k(sum(t in MARGIN_DN for t in toks+bigr),L),
            "capex_invest_k": per_k(sum(t in CAPEX_INV for t in toks+bigr),L),
            "ai_invest_k": per_k(sum(t in AI_INV for t in toks+bigr),L),
            "buyback_k": per_k(sum(t in BUYBACK for t in toks+bigr),L),
            "dividend_up_k": per_k(sum(t in DIV_UP for t in toks+bigr),L),
            "dividend_down_k": per_k(sum(t in DIV_DN for t in toks+bigr),L),
            "layoffs_k": per_k(sum(t in LAYOFFS for t in toks+bigr),L),
            "hiring_k": per_k(sum(t in HIRING for t in toks+bigr),L)
        }
        feats.append(f)
        hashes.append(hash_row(row))

    txt_feats = pd.DataFrame(feats)
    reports["hash"] = hashes

    print("[3/5] Computing embeddings with caching")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
            print("    -> Using GPU for embeddings")
        except Exception as e:
            print("    -> GPU fallback to CPU:", str(e))
    else:
        print("    -> No GPU detected, running on CPU")

    if os.path.exists(EMB_CACHE):
        cache_df = pd.read_parquet(EMB_CACHE)
    else:
        cache_df = pd.DataFrame(columns=["hash"]+[f"dim{i}" for i in range(384)])

    cached_hashes = set(cache_df["hash"].tolist())
    new_rows = reports[~reports["hash"].isin(cached_hashes)]

    if not new_rows.empty:
        new_embs = model.encode(new_rows["text"].astype(str).tolist(), show_progress_bar=True)
        new_embs_df = pd.DataFrame(new_embs, columns=[f"dim{i}" for i in range(new_embs.shape[1])])
        new_embs_df.insert(0, "hash", new_rows["hash"].values)
        cache_df = pd.concat([cache_df, new_embs_df], ignore_index=True).drop_duplicates("hash")
        cache_df.to_parquet(EMB_CACHE, index=False)

    emb_dict = {row["hash"]: row[1:].values for _,row in cache_df.iterrows()}
    embs = np.stack([emb_dict[h] for h in reports["hash"]])
    prev = reports.groupby("id").cumcount()-1
    cos = np.zeros(len(reports))
    for i in range(len(reports)):
        if prev[i]>=0:
            cos[i]=(embs[i]@embs[prev[i]])/(norm(embs[i])*norm(embs[prev[i]]))
    txt_feats["novelty_score"] = 1.0-cos

    print("[4/5] Merging text features into baseline dataset")
    reports["ret_eom"]=reports["char_eom"]+pd.DateOffset(months=4)
    df = base.merge(pd.concat([reports[["id","ret_eom"]], txt_feats], axis=1),
                    on=["id","ret_eom"], how="left")

    print("[5/5] Applying monthly z-scoring and saving output")
    def zfun(g):
        for c in txt_feats.columns:
            g[c]=(g[c]-g[c].mean())/g[c].std(ddof=0) if g[c].std(ddof=0)>0 else 0.0
        return g
    df = df.groupby("date", group_keys=False).apply(zfun).fillna(0)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_parquet(OUT_FILE, index=False, compression="snappy")
    print(f"    -> [DONE] wrote {OUT_FILE} with shape {df.shape}")

if __name__=="__main__":
    main()

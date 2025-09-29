"""
nlp_engine.py
Extract advanced text-based financial factors and merge into baseline dataset.
Steps:
  1. Load baseline & reports
  2. Compute lexicon-based sentiment/risk features
  3. Generate novelty score using embeddings
  4. Merge with baseline dataset
  5. Apply monthly z-scoring
"""

import os, re, argparse
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

BASELINE_FILE = "output/data/final_data.csv"
OUT_FILE      = "output/data/final_data_text.csv"
CACHE_DIR     = "output/cache"

# --- Lexicons ---
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_csv", type=str, default=None)
    ap.add_argument("--text_date_col", type=str, default="char_eom")
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[1/5] Loading baseline data")
    base = pd.read_csv(BASELINE_FILE, parse_dates=["date","ret_eom"], low_memory=False)

    if not args.reports_csv or not os.path.exists(args.reports_csv):
        base.to_csv(OUT_FILE, index=False)
        print(f"    -> No reports found, baseline copied to {OUT_FILE}")
        return

    print("[2/5] Loading reports and computing lexicon features")
    txt = pd.read_csv(args.reports_csv, low_memory=False)
    txt[args.text_date_col] = pd.to_datetime(txt[args.text_date_col], errors="coerce")

    feats = []
    for _,row in txt.iterrows():
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

    txt_feats = pd.DataFrame(feats)

    print("[3/5] Computing embeddings for novelty score")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if args.use_gpu:
        try: model = model.to("cuda")
        except: pass
    txt = txt.sort_values(["id", args.text_date_col]).reset_index(drop=True)
    embs = model.encode(txt["text"].astype(str).tolist(), show_progress_bar=True)
    prev = txt.groupby("id").cumcount()-1
    cos = np.zeros(len(txt))
    for i in range(len(txt)):
        if prev[i]>=0:
            cos[i]=(embs[i]@embs[prev[i]])/(norm(embs[i])*norm(embs[prev[i]]))
    txt_feats["novelty_score"] = 1.0-cos

    print("[4/5] Merging text features into baseline dataset")
    if args.text_date_col=="char_eom":
        txt["ret_eom"]=txt["char_eom"]+pd.DateOffset(months=4)
    else:
        txt["ret_eom"]=txt[args.text_date_col]
    df = base.merge(pd.concat([txt[["id","ret_eom"]], txt_feats], axis=1),
                    on=["id","ret_eom"], how="left")

    print("[5/5] Applying monthly z-scoring and saving output")
    def zfun(g):
        for c in txt_feats.columns:
            g[c]=(g[c]-g[c].mean())/g[c].std(ddof=0) if g[c].std(ddof=0)>0 else 0.0
        return g
    df = df.groupby("date", group_keys=False).apply(zfun).fillna(0)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"    -> [DONE] wrote {OUT_FILE} with shape {df.shape}")

if __name__=="__main__": 
    main()

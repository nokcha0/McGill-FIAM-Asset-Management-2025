# train_dualengine.py
# LightGBM regressor combining lead-4 accounting ratios + text-derived features

import os, argparse, datetime
import numpy as np, pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import pyarrow.parquet as pq

IN_FILE  = "output/data/final_data_text.parquet"
OUT_PRED = "output/predictions/dualengine.csv"
ACC_FILE = "/teamspace/uploads/acc_ratios.csv"

TEXT_COLS = [
    "sent_pos_k","sent_neg_k","sent_net_k","uncertainty_k",
    "risk_reg_k","risk_legal_k","risk_supply_k","risk_cyber_k","risk_macro_k","risk_geo_k",
    "cost_pressure_k","margin_up_k","margin_down_k","capex_invest_k","ai_invest_k",
    "buyback_k","dividend_up_k","dividend_down_k","layoffs_k","hiring_k","novelty_score"
]

def get_year(s):
    return pd.to_datetime(s).dt.year

def rank_scale_monthly(df, cols, date_col="date"):
    def _xform(g):
        g = g.copy()
        for c in cols:
            med = g[c].median(skipna=True)
            g[c] = g[c].fillna(med)
            r = g[c].rank(method="dense") - 1
            vmax = r.max()
            g[c] = (r / vmax) * 2 - 1 if vmax > 0 else 0.0
        return g
    return df.groupby(date_col, group_keys=False).apply(_xform)

def make_folds(dates, min_train_years=5):
    years = sorted(get_year(dates).unique())
    folds = []
    for k in range(min_train_years + 2, len(years)):
        y_test, y_val = years[k], years[k-1]
        train_years = [y for y in years if y <= y_val - 1]
        if len(train_years) >= min_train_years:
            folds.append((train_years, y_val, y_test))
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
    ap.add_argument("--num_boost_round", type=int, default=400)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--min_train_years", type=int, default=5)
    args = ap.parse_args()

    print("=== Train Dual-Engine (LightGBM + NLP features) ===")
    print("Start:", datetime.datetime.now())
    os.makedirs("output/predictions", exist_ok=True)

    print("[1/5] Loading merged dataset (Parquet, column-pruned)")
    schema_cols = set(pq.ParquetFile(IN_FILE).schema.names)
    keep_cols = ["id","date","ret_eom","stock_ret"] \
                + [c for c in schema_cols if c.endswith("_lead4")] \
                + [c for c in TEXT_COLS if c in schema_cols]
    df = pd.read_parquet(IN_FILE, columns=keep_cols)
    for dc in ("date","ret_eom"):
        if dc in df.columns and not np.issubdtype(df[dc].dtype, np.datetime64):
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
    if "stock_ret" not in df.columns:
        raise ValueError("Target 'stock_ret' missing")

    print("[2/5] Selecting predictors")
    ratios = pd.read_csv(ACC_FILE)["Variable"].astype(str).tolist()
    lead4_cols = [f"{v}_lead4" for v in ratios if f"{v}_lead4" in df.columns]
    text_cols  = [c for c in TEXT_COLS if c in df.columns]
    predictors = lead4_cols + text_cols
    if not predictors:
        raise ValueError("No predictors found")

    print("[3/5] Scaling features + creating folds")
    df = df.loc[df["stock_ret"].notna()].copy()
    df = rank_scale_monthly(df, predictors, date_col="date")
    folds = make_folds(df["date"], min_train_years=args.min_train_years)
    if not folds:
        raise RuntimeError("Not enough years for folds")

    pred_chunks = []

    print("[4/5] Training LightGBM across folds")
    for train_years, y_val, y_test in tqdm(folds, desc="Processing folds", unit="fold"):
        tr = df[get_year(df["date"]).isin(train_years)]
        va = df[get_year(df["date"]) == y_val]
        te = df[get_year(df["date"]) == y_test]
        if tr.empty or va.empty or te.empty:
            continue

        X_tr, y_tr = tr[predictors], tr["stock_ret"]
        X_va, y_va = va[predictors], va["stock_ret"]
        X_te, y_te = te[predictors], te["stock_ret"]

        params = {
            "objective": "regression", "metric": "l2", "boosting_type": "gbdt",
            "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1
        }
        if not args.cpu:
            params.update({"device": "gpu"})

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dtrain, dvalid], valid_names=["train","valid"],
            early_stopping_rounds=args.early_stopping_rounds, verbose_eval=False
        )

        yhat_te = model.predict(X_te, num_iteration=model.best_iteration)
        chunk = te[["id","ret_eom"]].copy()
        chunk["y_pred"] = yhat_te
        pred_chunks.append(chunk)

    if not pred_chunks:
        raise RuntimeError("No predictions produced")

    preds = pd.concat(pred_chunks).sort_values(["ret_eom","id"]).reset_index(drop=True)
    preds.to_csv(OUT_PRED, index=False)
    print("    -> Saved predictions:", OUT_PRED)

    print("[5/5] Evaluating out-of-sample performance")
    target = df[["id","ret_eom","stock_ret"]].copy()
    oos = preds.merge(target, on=["id","ret_eom"], how="left")
    if oos["stock_ret"].notna().any():
        y, yhat = oos["stock_ret"].values, oos["y_pred"].values
        r2 = 1 - np.sum((y - yhat)**2) / np.sum(y**2) if np.sum(y**2) > 0 else np.nan
        print("    -> Dual-engine OOS R^2:", round(float(r2), 4))

    print("End:", datetime.datetime.now())

if __name__ == "__main__":
    main()

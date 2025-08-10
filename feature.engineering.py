# feature.engineering.py

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_BASE_CANDIDATES = [
    Path("/Users/julianfrost/Documents/PSP_Analysis/data"),
    Path("/mnt/data"),
    Path.cwd() / "data",
]
MASTER_NAME = "master.csv"
FEATURES_NAME = "features.csv"

PRICES = ["spy_price", "msft_price", "eth_price", "oil_price"]
ACTIVITY = ["spy_earnings", "msft_earnings", "eth_fees", "oil_consumption"]
RET_HORIZONS = (1, 5, 10)
ACT_LAGS = (1, 3, 5, 10)
ROLL_WINDOW = 504  # ≈ 2y of business days


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_base(user_base: str | None) -> Path:
    if user_base:
        p = Path(user_base).expanduser().resolve()
        if not p.exists():
            sys.exit(f"Base path not found: {p}")
        return p
    for cand in DEFAULT_BASE_CANDIDATES:
        if cand.exists():
            return cand
    sys.exit("Could not locate a valid data directory. Use --base to specify one.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, help="Directory containing master.csv and where features.csv will be written")
    args = ap.parse_args()

    base = pick_base(args.base)
    master_fp = base / MASTER_NAME
    if not master_fp.exists():
        sys.exit(f"Could not find {master_fp}")

    # 1) Load master panel
    df = pd.read_csv(master_fp, parse_dates=["date"], index_col="date").sort_index()
    have_cols = set(df.columns)

    # Sanity: ensure required columns exist
    missing_prices = [c for c in PRICES if c not in have_cols]
    missing_activity = [c for c in ACTIVITY if c not in have_cols]
    if missing_prices or missing_activity:
        sys.exit(f"Missing columns in master.csv.\n"
                 f"   Prices missing: {missing_prices}\n"
                 f"   Activity missing: {missing_activity}")

    # 2) Forward log-returns (per asset)
    for h in RET_HORIZONS:
        for p in PRICES:
            df[f"{p}_ret{h}d"] = np.log(df[p].shift(-h) / df[p])

    # AR(1) of the dependent variable (lag of 1d forward return)
    for p in PRICES:
        col = f"{p}_ret1d"
        df[f"{col}_lag1"] = df[col].shift(1)

    # 3) Activity %-changes at appropriate cadences
    for a in ACTIVITY:
        if a.endswith("_earnings"):
            # Earnings series are quarterly but daily-forward-filled: year-over-year change
            df[f"d_{a}"] = df[a].pct_change(252)
        elif a == "oil_consumption":
            # Monthly but daily-forward-filled: approx month-over-month change
            df[f"d_{a}"] = df[a].pct_change(21)
        else:
            # Daily (ETH fees)
            df[f"d_{a}"] = df[a].pct_change(1)

    # 4) Lags of activity levels (not changes)
    for a in ACTIVITY:
        for L in ACT_LAGS:
            df[f"{a}_lag{L}"] = df[a].shift(L)

    # 5) Shock flags (top-decile of Δ over 2y rolling window)
    for a in ACTIVITY:
        delta = df[f"d_{a}"]
        q = delta.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW // 2).quantile(0.90)
        df[f"{a}_shock"] = (delta > q).astype(int)

# 6) Rate-regime dummies (optional, only if GS3M exists)
    if "GS3M" in df.columns:
        # Terciles of 3M yield; make sure duplicate cutpoints don't crash
        bins = pd.qcut(
            df["GS3M"], 3,
            labels=["regime_low", "regime_mid", "regime_high"],
            duplicates="drop"
        )
        dummies = pd.get_dummies(bins, dtype=int)

        # Force a plain object Index on columns to avoid CategoricalIndex join bug
        dummies = pd.DataFrame(
            dummies.to_numpy(),
            index=df.index,
            columns=[str(c) for c in dummies.columns]
        )
        df = df.join(dummies)
        print("✔ Added GS3M rate-regime dummies")
    else:
        print("↪ Skipping rate-regime dummies (GS3M not found in master.csv)")


    # 7) Final cleanup & save
    # Keep rows where 1d forward returns exist for all assets (so next-day targets are defined)
    target_cols = [f"{p}_ret1d" for p in PRICES]
    before = len(df)
    df = df.dropna(subset=target_cols)
    after = len(df)

    out_fp = base / FEATURES_NAME
    df.to_csv(out_fp, index=True)
    print(
        f"✓ features.csv saved: {out_fp}\n"
        f"  rows={after} (dropped {before - after} rows without next-day returns), "
        f"cols={df.shape[1]}, "
        f"range={df.index.min().date()} → {df.index.max().date()}"
    )


if __name__ == "__main__":
    main()

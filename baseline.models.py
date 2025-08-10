# baseline.models.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_BASE_CANDIDATES = [
    Path("/Users/julianfrost/Documents/PSP_Analysis/data"),
    Path("/mnt/data"),
    Path.cwd() / "data",
]
FEATURES_NAME = "features.csv"
RESULTS_NAME = "baseline_results.csv"

ASSETS = {
    "SPY":  "spy_price_ret1d",
    "MSFT": "msft_price_ret1d",
    "ETH":  "eth_price_ret1d",
    "OIL":  "oil_price_ret1d",
}
ACTIVITY = [
    "d_spy_earnings",  "spy_earnings_lag1",
    "d_msft_earnings", "msft_earnings_lag1",
    "d_eth_fees",      "eth_fees_lag1",
    "d_oil_consumption","oil_consumption_lag1",
]
RATE_CANDIDATES = ["GS3M", "GS10", "regime_low", "regime_mid", "regime_high"]


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


def available_rates(df: pd.DataFrame) -> list[str]:
    """Return rate/regime columns that exist; drop one regime dummy to avoid the trap."""
    present = [c for c in RATE_CANDIDATES if c in df.columns]
    regimes = [c for c in present if c.startswith("regime_")]
    if len(regimes) >= 1:
        # Always drop exactly one regime dummy when intercept is present
        # (if 3 exist, drop 'regime_low'; if only 2, drop the first lexicographically)
        drop = "regime_low" if "regime_low" in regimes else sorted(regimes)[0]
        present.remove(drop)
    return present


def fit_ols_hac(y: pd.Series, X: pd.DataFrame, hac_lags: int):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop")
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def collect_results(asset: str, bundle: str, mdl, start, end) -> pd.DataFrame:
    out = pd.DataFrame({
        "variable": mdl.params.index,
        "coef": mdl.params.values,
        "std_err": mdl.bse.values,
        "t": mdl.tvalues.values,
        "p": mdl.pvalues.values,
    })
    out["asset"] = asset
    out["bundle"] = bundle
    out["r2"] = mdl.rsquared
    out["r2_adj"] = mdl.rsquared_adj
    out["n"] = int(mdl.nobs)
    out["start"] = str(start.date()) if pd.notna(start) else None
    out["end"] = str(end.date()) if pd.notna(end) else None
    cols = ["asset","bundle","variable","coef","std_err","t","p","r2","r2_adj","n","start","end"]
    return out[cols]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, help="Directory containing features.csv; results will be written here too")
    ap.add_argument("--hac_lags", type=int, default=5, help="Newey–West maxlags (default: 5)")
    args = ap.parse_args()

    base = pick_base(args.base)
    feats_fp = base / FEATURES_NAME
    if not feats_fp.exists():
        sys.exit(f"Could not find {feats_fp}")

    df = pd.read_csv(feats_fp, parse_dates=["date"], index_col="date").sort_index()

    # Verify targets
    missing_targets = [v for v in ASSETS.values() if v not in df.columns]
    if missing_targets:
        sys.exit(f"Missing target columns in features.csv: {missing_targets}")

    # Determine available controls
    rates = available_rates(df)
    have_rates = len(rates) > 0

    # Define bundles dynamically
    bundle_specs = {
        "activity": list(ACTIVITY),
    }
    if have_rates:
        bundle_specs["act+rates"] = list(ACTIVITY) + rates
    # act+rates+ar: if rates present, add them; always add AR(1)
    for name in ["act+rates+ar"]:
        base_cols = list(ACTIVITY) + (rates if have_rates else [])
        bundle_specs[name] = base_cols  # AR(1) added per-asset below

    rows = []

    for bundle, Xbase in bundle_specs.items():
        for asset, ycol in ASSETS.items():
            Xcols = list(Xbase)
            if bundle == "act+rates+ar":
                ar_col = f"{ycol}_lag1"   # e.g., spy_price_ret1d_lag1
                if ar_col in df.columns:
                    Xcols.append(ar_col)
                else:
                    print(f"↪ Warning: {ar_col} not found; proceeding without AR(1) for {asset}")

            # Keep only existing columns (defensive)
            Xcols = [c for c in Xcols if c in df.columns]

            sub = df[[ycol] + Xcols].dropna()
            if sub.empty or sub.shape[0] < 60:
                print(f"↪ Skipping {asset} | {bundle}: not enough rows after dropna (rows={len(sub)})")
                continue

            y = sub[ycol]
            X = sub[Xcols]

            try:
                mdl = fit_ols_hac(y, X, hac_lags=args.hac_lags)
            except Exception as e:
                print(f"↪ Skipping {asset} | {bundle} due to OLS error: {e}")
                continue

            start, end = sub.index.min(), sub.index.max()
            res = collect_results(asset, bundle, mdl, start, end)
            rows.append(res)

            print(f"\n=== {asset} | {bundle} | R²={mdl.rsquared:.3f} | N={int(mdl.nobs)} ===")
            print(res[res["variable"].isin(["const"] + Xcols)][["variable","coef","p"]].head(10).to_string(index=False))

    if not rows:
        sys.exit("No models were fit; check features and bundle specs.")

    out = pd.concat(rows, ignore_index=True)
    out_fp = base / RESULTS_NAME
    out.to_csv(out_fp, index=False)
    print(f"\n✔ {RESULTS_NAME} saved: {out_fp}  shape={out.shape}")


if __name__ == "__main__":
    main()

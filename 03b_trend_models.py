# 03b_trend_models.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_BASE_CANDIDATES = [
    Path("/Users/julianfrost/Documents/PSP_Analysis/data"),
    Path("/mnt/data"),
    Path.cwd() / "data",
]

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

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"features.csv missing columns: {missing}")

def add_earnings_trends(df: pd.DataFrame) -> pd.DataFrame:
    # Quarter ≈ 63 business days
    windows = {"2q": 126, "3q": 189, "4q": 252}
    for base in ["spy_earnings", "msft_earnings"]:
        if base not in df.columns:
            continue
        for tag, w in windows.items():
            df[f"d_{base}_{tag}"] = df[base].pct_change(w)
    return df

def nw_ols(y: pd.Series, X: pd.DataFrame, maxlags: int = 5):
    X = sm.add_constant(X, has_constant="add")
    mdl = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return mdl

def part_a_earnings_trends(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure we have needed columns
    ensure_cols(df, ["spy_price_ret1d","msft_price_ret1d","GS3M","GS10"])
    df = add_earnings_trends(df)

    rows = []
    specs = [
        ("SPY", "spy_price_ret1d",  "spy_earnings"),
        ("MSFT","msft_price_ret1d", "msft_earnings"),
    ]
    trend_suffixes = ["2q","3q","4q"]

    for asset, ycol, base in specs:
        for suf in trend_suffixes:
            trend = f"d_{base}_{suf}"
            if trend not in df.columns:
                continue
            Xcols = [trend, f"{ycol.replace('_ret1d','')}_lag1", "GS3M", "GS10"]
            Xcols = [c for c in Xcols if c in df.columns]
            sub = df[[ycol] + Xcols].dropna()
            if len(sub) < 200:
                continue
            mdl = nw_ols(sub[ycol], sub[Xcols], maxlags=5)
            rows.append({
                "asset": asset,
                "trend_var": trend,
                "coef_trend": mdl.params.get(trend, np.nan),
                "se_trend":   mdl.bse.get(trend, np.nan),
                "t_trend":    mdl.tvalues.get(trend, np.nan),
                "p_trend":    mdl.pvalues.get(trend, np.nan),
                "r2":         mdl.rsquared,
                "n":          int(mdl.nobs)
            })
            print(f"{asset} ~ {trend}: coef={mdl.params.get(trend, np.nan):.4e} "
                  f"(p={mdl.pvalues.get(trend, np.nan):.3f})  R²={mdl.rsquared:.3f}  N={int(mdl.nobs)}")
    return pd.DataFrame(rows)

def monthly_log_return(s: pd.Series) -> pd.Series:
    # monthly log return: last/first within month
    m = s.resample("M")
    return np.log(m.last() / m.first())

def part_b_rates_monthly(df: pd.DataFrame) -> pd.DataFrame:
    # Need price levels for monthly returns + rate levels
    need_any = ["spy_price","msft_price","GS3M","GS10"]
    ensure_cols(df, [c for c in need_any if c in df.columns])

    out_rows = []
    # Monthly returns
    series = {}
    if "spy_price" in df.columns:
        series["SPY"]  = monthly_log_return(df["spy_price"].dropna())
    if "msft_price" in df.columns:
        series["MSFT"] = monthly_log_return(df["msft_price"].dropna())

    # Monthly Δrates (levels, not %)
    d_gs3m = df["GS3M"].resample("M").last().diff()
    d_gs10 = df["GS10"].resample("M").last().diff()

    for asset, r_m in series.items():
        tmp = pd.concat([r_m, d_gs3m, d_gs10], axis=1, keys=["ret_m","d_gs3m","d_gs10"]).dropna()
        if len(tmp) < 36:
            continue

        # Correlations
        corr_3m  = float(tmp["ret_m"].corr(tmp["d_gs3m"]))
        corr_10y = float(tmp["ret_m"].corr(tmp["d_gs10"]))

        # OLS: ret_m ~ Δrate  (one at a time)
        for var in ["d_gs3m","d_gs10"]:
            X = sm.add_constant(tmp[[var]], has_constant="add")
            mdl = sm.OLS(tmp["ret_m"], X).fit(cov_type="HAC", cov_kwds={"maxlags":3})
            out_rows.append({
                "asset": asset, "horizon": "monthly",
                "predictor": var,
                "coef": mdl.params.get(var, np.nan),
                "se": mdl.bse.get(var, np.nan),
                "t": mdl.tvalues.get(var, np.nan),
                "p": mdl.pvalues.get(var, np.nan),
                "r2": mdl.rsquared,
                "n": int(mdl.nobs),
                "corr": corr_3m if var=="d_gs3m" else corr_10y
            })
            print(f"[Monthly] {asset} ~ {var}: coef={mdl.params.get(var, np.nan):.4e} "
                  f"(p={mdl.pvalues.get(var, np.nan):.3f})  corr={out_rows[-1]['corr']:.3f}  N={int(mdl.nobs)}")

    return pd.DataFrame(out_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str)
    args = ap.parse_args()

    base = pick_base(args.base)
    feats_fp = base / "features.csv"
    if not feats_fp.exists():
        sys.exit(f"Could not find {feats_fp}")

    df = pd.read_csv(feats_fp, parse_dates=["date"], index_col="date").sort_index()

    # Part A: earnings multi-period trends
    trends = part_a_earnings_trends(df)
    out_a = base / "baseline_trends.csv"
    trends.to_csv(out_a, index=False)
    print(f"✓ baseline_trends.csv written: {out_a}  shape={trends.shape}")

    # Part B: monthly rate/return relations
    rates = part_b_rates_monthly(df)
    out_b = base / "rates_trends_monthly.csv"
    rates.to_csv(out_b, index=False)
    print(f"✓ rates_trends_monthly.csv written: {out_b}  shape={rates.shape}")

if __name__ == "__main__":
    main()



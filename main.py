# main.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# soft import for HAC regression used in the ETH same-day check
try:
    import statsmodels.api as sm
except Exception:
    sm = None


# ---------------------------------------------------------------------
# Defaults & helpers
# ---------------------------------------------------------------------
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

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def exp_decay(t: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    return alpha * np.exp(-delta * t)

def read_csv_if_exists(fp: Path, **kwargs) -> pd.DataFrame | None:
    if fp.exists():
        return pd.read_csv(fp, **kwargs)
    return None

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace({"": None}),
        errors="coerce"
    )

def load_simple_series(fp: Path, date_col: str = "Date") -> pd.Series:
    """
    Minimal loader for 2-column csvs like eth_price.csv / eth_fees.csv.
    Returns a Series with DatetimeIndex and float values.
    """
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    if date_col not in df.columns or df.shape[1] < 2:
        raise ValueError(f"{fp} needs at least columns: {date_col}, <value>")
    val_col = df.columns[1]
    df = df.rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["value"] = coerce_numeric(df["value"])
    return pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"].values), name=fp.stem)

def choose_preferred_row(rob: pd.DataFrame,
                         asset: str,
                         pctl_target: float = 0.90,
                         resid_target: str = "ar+rates",
                         sign_target: str = "all") -> pd.Series | None:
    """
    Pick preferred robustness row with graceful fallback:
      (asset match) & (sign=sign_target) & (residualize=resid_target) & (pctl closest to pctl_target)
    Fallbacks relax residualize→any, sign→any, then pick closest pctl.
    """
    if rob is None or rob.empty:
        return None
    df = rob.copy()
    df = df[df["asset"] == asset]
    if df.empty:
        return None

    def pick(d: pd.DataFrame, require_resid: bool, require_sign: bool) -> pd.Series | None:
        tmp = d
        if require_resid:
            tmp = tmp[tmp["residualize"] == resid_target]
        if require_sign:
            tmp = tmp[tmp.get("sign", "all") == sign_target]
        if tmp.empty:
            return None
        tmp = tmp.assign(_dist=(tmp["pctl"] - pctl_target).abs())
        tmp = tmp.sort_values(["_dist"]).drop(columns=["_dist"])
        return tmp.iloc[0]

    # try strict, then relax residualize, then relax sign
    row = pick(df, True, True)
    if row is None: row = pick(df, False, True)
    if row is None: row = pick(df, False, False)
    return row


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def build_baseline_table(base: Path) -> Path | None:
    src = base / "baseline_results.csv"
    tbl = read_csv_if_exists(src)
    if tbl is None:
        print("⚠️  baseline_results.csv not found; skipping baseline table.")
        return None

    # Normalize common column names
    cols = {c.lower(): c for c in tbl.columns}
    # Expected: ["variable","Coef.","Std.Err.","P>|t|","asset","bundle", ...]
    var_col = next((c for c in tbl.columns if c.lower() == "variable"), None)
    coef_col = next((c for c in tbl.columns if c.lower().startswith("coef")), None)
    se_col   = next((c for c in tbl.columns if "std" in c.lower() and "err" in c.lower()), None)
    p_col    = next((c for c in tbl.columns if c.lower().startswith("p")), None)
    if not all([var_col, coef_col, se_col, p_col]) or "asset" not in tbl.columns or "bundle" not in tbl.columns:
        print("⚠️  baseline_results.csv missing expected columns; saving raw file as-is.")
        out = base / "table_baseline_activity.csv"
        tbl.to_csv(out, index=False)
        return out

    # Focus on activity predictors only (clean, readable table)
    activity_vars = [
        "d_spy_earnings", "spy_earnings_lag1",
        "d_msft_earnings","msft_earnings_lag1",
        "d_eth_fees",     "eth_fees_lag1",
        "d_oil_consumption","oil_consumption_lag1"
    ]
    keep = tbl[tbl[var_col].isin(activity_vars)].copy()
    keep = keep[["asset","bundle",var_col,coef_col,se_col,p_col]].rename(columns={
        var_col: "variable", coef_col: "coef", se_col: "se", p_col: "p"
    })
    # optional stars
    keep["stars"] = pd.cut(keep["p"], bins=[-1,0.01,0.05,0.10,1.01], labels=["***","**","*",""], right=True)
    out = base / "table_baseline_activity.csv"
    keep.to_csv(out, index=False)
    print(f"✓ Saved: {out}  shape={keep.shape}")
    return out


def build_branching_tables_and_plots(base: Path,
                                     irf_dir: Path,
                                     figs_dir: Path,
                                     pctl: float,
                                     resid: str,
                                     sign: str) -> tuple[Path | None, Path | None]:
    single = read_csv_if_exists(base / "branching_params.csv")
    robust = read_csv_if_exists(base / "branching_params_robust.csv")

    if single is not None and not single.empty:
        out_single = base / "table_branching_single.csv"
        single.to_csv(out_single, index=False)
        print(f"✓ Saved: {out_single}  shape={single.shape}")
    else:
        out_single = None
        print("⚠️  branching_params.csv not found or empty; skipping single table.")

    # Preferred per asset from robust (fall back to single if needed)
    assets = ["SPY","MSFT","ETH","OIL"]
    rows = []
    for a in assets:
        row = None
        if robust is not None and not robust.empty:
            row = choose_preferred_row(robust, a, pctl_target=pctl, resid_target=resid, sign_target=sign)
        if row is None and single is not None and not single.empty:
            row = single[single["asset"] == a].iloc[0]
        if row is not None:
            rows.append(row)

    if rows:
        pref = pd.DataFrame(rows).copy()
        out_pref = base / "table_branching_preferred.csv"
        pref.to_csv(out_pref, index=False)
        print(f"✓ Saved: {out_pref}  shape={pref.shape}")
        # Plot IRFs with fitted curve
        ensure_dir(figs_dir)
        for a in assets:
            # Try to read preferred IRF first (robust tag), else fall back to single IRF
            tag_path = irf_dir / f"irf_{a.lower()}__p{int(round(pctl*100))}__resid_{resid}__sign_{sign}.csv"
            irf_fp = tag_path if tag_path.exists() else (irf_dir / f"irf_{a.lower()}.csv")
            irf = read_csv_if_exists(irf_fp)
            if irf is None or irf.empty:
                print(f"⚠️  No IRF file for {a} at {irf_fp}; skipping plot.")
                continue

            # Get params for overlay
            if robust is not None and not robust.empty:
                r = choose_preferred_row(robust, a, pctl_target=pctl, resid_target=resid, sign_target=sign)
            else:
                r = single[single["asset"] == a].iloc[0] if single is not None and not single.empty else None

            t = irf["t"].values
            y = irf["cum_ret"].values

            plt.figure(figsize=(6,4))
            plt.plot(t, y, label="IRF (avg cum. log-return)")
            if r is not None and np.isfinite(r.get("alpha", np.nan)) and np.isfinite(r.get("delta", np.nan)):
                # Fit in our estimation used t starting at 1; adapt overlay accordingly
                t_fit = (t - t.min()) + 1.0
                yhat = exp_decay(t_fit, float(r["alpha"]), float(r["delta"]))
                plt.plot(t, yhat, linestyle="--", label=f"fit: α={r['alpha']:.2e}, δ={r['delta']:.3f}")
            plt.axhline(0, linewidth=0.8)
            plt.title(f"{a} IRF (preferred spec)")
            plt.xlabel("event time t")
            plt.ylabel("cum. log-return")
            plt.legend()
            out_png = figs_dir / f"irf_{a.lower()}_preferred.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"✓ Figure: {out_png}")
    else:
        out_pref = None
        print("⚠️  No preferred branching rows found; skipping plots.")

    return out_single, out_pref


# ---------------------------------------------------------------------
# ETH same-day check (calendar days)
# ---------------------------------------------------------------------
def eth_same_day_check(base: Path, figs_dir: Path) -> tuple[Path | None, Path | None]:
    price_fp = base / "eth_price.csv"
    fees_fp  = base / "eth_fees.csv"
    try:
        p = load_simple_series(price_fp)   # price level
        f = load_simple_series(fees_fp)    # fee level
    except Exception as e:
        print(f"⚠️  ETH same-day check skipped: {e}")
        return None, None

    # Align on union of calendar days; compute same-day return and fee delta
    df = pd.DataFrame({"price": p, "fees": f}).sort_index()
    df["ret1d"] = np.log(df["price"] / df["price"].shift(1))
    df["d_fees"] = df["fees"].pct_change(1)
    df = df.dropna(subset=["ret1d", "d_fees"])
    if df.empty:
        print("⚠️  ETH same-day check: no overlapping data.")
        return None, None

    # Regress r_t ~ Δfees_t (HAC Newey-West SE). If statsmodels missing, save raw corr.
    out_csv = base / "eth_same_day_reg.csv"
    out_png = figs_dir / "eth_same_day_scatter.png"
    ensure_dir(figs_dir)

    if sm is None:
        corr = df["ret1d"].corr(df["d_fees"])
        pd.DataFrame([{"coef_d_fees": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                       "r2": corr**2, "n": len(df)}]).to_csv(out_csv, index=False)
        print(f"⚠️  statsmodels not available; saved correlation-only: {out_csv}")
    else:
        X = sm.add_constant(df["d_fees"], has_constant="add")
        mdl = sm.OLS(df["ret1d"], X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags":5})
        row = pd.DataFrame([{
            "coef_const": mdl.params.get("const", np.nan),
            "coef_d_fees": mdl.params.get("d_fees", np.nan),
            "se_d_fees": mdl.bse.get("d_fees", np.nan),
            "t_d_fees": mdl.tvalues.get("d_fees", np.nan),
            "p_d_fees": mdl.pvalues.get("d_fees", np.nan),
            "r2": mdl.rsquared,
            "n": int(mdl.nobs)
        }])
        row.to_csv(out_csv, index=False)
        print(f"✓ Saved: {out_csv}  n={int(mdl.nobs)} R²={mdl.rsquared:.3f}")

    # Scatter + OLS line for visualization (if possible)
    plt.figure(figsize=(6,4))
    plt.scatter(df["d_fees"], df["ret1d"], s=10, alpha=0.6)
    if sm is not None:
        b0 = row["coef_const"].iloc[0]
        b1 = row["coef_d_fees"].iloc[0]
        xs = np.linspace(df["d_fees"].quantile(0.01), df["d_fees"].quantile(0.99), 100)
        plt.plot(xs, b0 + b1*xs, linestyle="--", linewidth=1.5)
    plt.title("ETH same-day: return vs Δfees")
    plt.xlabel("Δ fees (1-day % change)")
    plt.ylabel("same-day log return")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✓ Figure: {out_png}")

    return out_csv, out_png


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str)
    ap.add_argument("--preferred_pctl", type=float, default=0.90)
    ap.add_argument("--preferred_resid", type=str, default="ar+rates")
    ap.add_argument("--preferred_sign", type=str, default="all")
    args = ap.parse_args()

    base = pick_base(args.base)
    irf_dir = base / "irfs"
    figs_dir = base / "figs"
    ensure_dir(figs_dir)

    print(f"📂 Using data dir: {base}")
    print(f"🖼  Figures -> {figs_dir}")

    # 1) Baseline regression table
    build_baseline_table(base)

    # 2) Branching tables & preferred plots
    build_branching_tables_and_plots(
        base=base,
        irf_dir=irf_dir,
        figs_dir=figs_dir,
        pctl=args.preferred_pctl,
        resid=args.preferred_resid,
        sign=args.preferred_sign
    )

    # 3) ETH same-day check (calendar days)
    eth_same_day_check(base, figs_dir)

    print("✅ main.py completed.")

if __name__ == "__main__":
    main()

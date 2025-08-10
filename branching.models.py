# branching.models.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------
# Defaults / metadata
# ---------------------------------------------------------------------
DEFAULT_BASE_CANDIDATES = [
    Path("/Users/julianfrost/Documents/PSP_Analysis/data"),
    Path("/mnt/data"),
    Path.cwd() / "data",
]
FEATURES_NAME = "features.csv"
PARAMS_SINGLE = "branching_params.csv"
PARAMS_ROBUST = "branching_params_robust.csv"

ASSET_CFG = {
    "SPY":  {"ret": "spy_price_ret1d",  "level": "spy_earnings",     "delta": "d_spy_earnings",    "flag": "spy_earnings_shock"},
    "MSFT": {"ret": "msft_price_ret1d", "level": "msft_earnings",    "delta": "d_msft_earnings",   "flag": "msft_earnings_shock"},
    "ETH":  {"ret": "eth_price_ret1d",  "level": "eth_fees",         "delta": "d_eth_fees",        "flag": "eth_fees_shock"},
    "OIL":  {"ret": "oil_price_ret1d",  "level": "oil_consumption",  "delta": "d_oil_consumption", "flag": "oil_consumption_shock"},
}

ROLL_WINDOW = 504  # ~2 trading years


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def exp_decay(t: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    return alpha * np.exp(-delta * t)  # δ bounded ≥0 in curve_fit


def winsorize(s: pd.Series, a: float) -> pd.Series:
    if a <= 0:
        return s
    lo, hi = s.quantile(a), s.quantile(1 - a)
    return s.clip(lo, hi)


def residualize(y: pd.Series, df: pd.DataFrame, ycol: str, mode: str) -> pd.Series:
    """Return residuals of y after regressing on requested controls (safe fallbacks)."""
    if mode == "none":
        return y

    controls: list[str] = []
    if mode in {"ar", "ar+rates"}:
        ar = f"{ycol}_lag1"
        if ar in df.columns:
            controls.append(ar)

    if mode in {"rates", "ar+rates"}:
        for c in ["GS3M", "GS10", "regime_mid", "regime_high", "regime_low"]:
            if c in df.columns:
                controls.append(c)
        if {"regime_low", "regime_mid", "regime_high"}.issubset(df.columns):
            try:
                controls.remove("regime_low")  # avoid dummy trap
            except ValueError:
                pass

    if not controls:
        return y

    sub = df[[ycol] + controls].dropna()
    if len(sub) < 100:
        return y  # not enough observations to stabilize residuals

    import statsmodels.api as sm  # local import to keep dependency soft
    X = sm.add_constant(sub[controls], has_constant="add")
    mdl = sm.OLS(sub[ycol], X).fit()
    r = y.copy()
    r.loc[sub.index] = mdl.resid
    return r


def shock_series_auto(df: pd.DataFrame, level_col: str, delta_col: str, two_sided: bool) -> pd.Series:
    """
    Auto shock driver:
      - Prefer 1d % change of the LEVEL series (captures the actual step day)
      - Fall back to the provided delta column if level is missing.
    """
    if level_col in df.columns:
        s = df[level_col].pct_change(1)
    elif delta_col in df.columns:
        s = df[delta_col]
    else:
        raise KeyError(f"Neither {level_col} nor {delta_col} present in features.csv")
    return s.abs() if two_sided else s


def threshold_mask(series: pd.Series, pctl: float, two_sided: bool, sign: str = "all") -> pd.Series:
    """
    Rolling top-pctl mask. If sign in {'pos','neg'}, filter by sign after
    thresholding on |series| when two_sided=True (or raw series otherwise).
    """
    base = series.abs() if two_sided else series
    q = base.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW // 2).quantile(pctl)
    m = (base > q).fillna(False)
    if sign == "pos":
        m = m & (series > 0)
    elif sign == "neg":
        m = m & (series < 0)
    return m


def first_of_run_with_cooldown(mask: pd.Series, idx: pd.DatetimeIndex, cooldown: int) -> pd.DatetimeIndex:
    """Take first day of each contiguous True-run, then enforce trading-day cooldown."""
    first = mask & (~mask.shift(1, fill_value=False))
    cand_dates = idx[first]
    picks: list[pd.Timestamp] = []
    last_pos = -10**9
    for d in cand_dates:
        pos = idx.get_loc(d)
        if pos - last_pos >= cooldown:
            picks.append(d)
            last_pos = pos
    return pd.DatetimeIndex(picks)


def build_irf(idx: pd.DatetimeIndex, ret: pd.Series,
              shocks: pd.DatetimeIndex, H: int, include_t0: bool) -> np.ndarray:
    """
    Return matrix (#shocks × H) of cumulative log-returns in event time.
    If include_t0, window is [t=0 .. H-1] starting at the shock day.
    Else, window is [t=1 .. H] starting the day after the shock.
    """
    rows: list[np.ndarray] = []
    for d in shocks:
        try:
            pos = idx.get_loc(d)
        except KeyError:
            continue
        start = pos if include_t0 else pos + 1
        end = start + H
        if end <= len(idx):
            w = ret.iloc[start:end].to_numpy()
            if np.isfinite(w).all() and len(w) == H:
                rows.append(np.cumsum(w))
    return np.vstack(rows) if rows else np.empty((0, H))


def fit_decay(irf_avg: np.ndarray) -> tuple[float, float, float]:
    """Fit α, δ with bounds (δ≥0). Return (alpha, delta, R²_fit)."""
    t = np.arange(0, len(irf_avg), dtype=float) + 1.0  # start at 1 for stability
    if (not np.isfinite(irf_avg).all()) or np.allclose(irf_avg, 0.0):
        return np.nan, np.nan, np.nan
    a0, d0 = float(irf_avg[0]), 0.2
    try:
        popt, _ = curve_fit(exp_decay, t, irf_avg,
                            p0=[a0, d0],
                            bounds=([-np.inf, 0.0], [np.inf, 5.0]),
                            maxfev=10000)
        alpha, delta = popt
        yhat = exp_decay(t, alpha, delta)
        sse = float(np.sum((irf_avg - yhat) ** 2))
        sst = float(np.sum((irf_avg - np.mean(irf_avg)) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        return alpha, delta, r2
    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------
def run_single(df: pd.DataFrame, irf_dir: Path, base: Path, horizon: int, pctl: float, cooldown: int,
               agg: str, winsor: float, two_sided: bool, residualize_mode: str,
               use_flags: bool, include_t0: bool) -> None:
    idx = df.index
    rows: list[dict] = []
    ensure_dir(irf_dir)

    for asset, cfg in ASSET_CFG.items():
        ycol, level_col, delta_col, flag_col = cfg["ret"], cfg["level"], cfg["delta"], cfg["flag"]
        if ycol not in df.columns:
            print(f"↪ Skipping {asset}: missing {ycol}")
            continue

        y = df[ycol].copy()
        y = residualize(y, df, ycol, residualize_mode)
        if agg == "winsor_mean":
            y = winsorize(y, winsor)

        # shocks
        if use_flags and (flag_col in df.columns):
            mask = df[flag_col].astype(int) == 1
        else:
            s = shock_series_auto(df, level_col, delta_col, two_sided)
            mask = threshold_mask(s, pctl=pctl, two_sided=two_sided, sign="all")
        shocks = first_of_run_with_cooldown(mask, idx, cooldown)

        # IRF
        mat = build_irf(idx, y, shocks, horizon, include_t0)
        n_total, n_used = int(mask.sum()), mat.shape[0]

        if n_used == 0:
            irf_avg = np.full(horizon, np.nan)
            alpha = delta = r2_fit = half = np.nan
        else:
            if agg == "median":
                irf_avg = np.nanmedian(mat, axis=0)
            else:
                irf_avg = np.nanmean(mat, axis=0)
            alpha, delta, r2_fit = fit_decay(irf_avg)
            half = (np.log(2) / delta) if (np.isfinite(delta) and delta > 0) else np.nan

        # save IRF to irfs/
        t0 = 0 if include_t0 else 1
        pd.DataFrame({"t": np.arange(t0, t0 + horizon), "cum_ret": irf_avg}) \
          .to_csv(irf_dir / f"irf_{asset.lower()}.csv", index=False)

        print(f"{asset}: shocks(total/used)={n_total}/{n_used}; "
              f"δ={delta if np.isfinite(delta) else np.nan:.4f}  "
              f"α={alpha if np.isfinite(alpha) else np.nan:.4e}  "
              f"R²_fit={r2_fit if np.isfinite(r2_fit) else np.nan:.3f}  "
              f"half={half if np.isfinite(half) else np.nan:.2f}d")

        rows.append({
            "asset": asset, "alpha": alpha, "delta": delta, "half_life_days": half, "fit_r2": r2_fit,
            "n_shocks_total": int(n_total), "n_shocks_used": int(n_used),
            "horizon": int(horizon), "pctl": float(pctl), "cooldown": int(cooldown),
            "agg": agg, "winsor": float(winsor), "two_sided": bool(two_sided),
            "residualize": residualize_mode, "include_t0": bool(include_t0),
            "start": str(idx.min().date()), "end": str(idx.max().date()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(base / PARAMS_SINGLE, index=False)
    print(f"✓ {PARAMS_SINGLE} written: {base / PARAMS_SINGLE}  shape={out.shape}")
    print(f"ℹ️ IRFs saved in: {irf_dir}")


# ---------------------------------------------------------------------
# Robust grid
# ---------------------------------------------------------------------
def run_robust(df: pd.DataFrame, irf_dir: Path, base: Path, horizon: int, cooldown: int,
               sign_split: bool) -> None:
    """
    Robust grid with fixed, defensible settings:
      pctl ∈ {0.85, 0.90, 0.95}
      residualize ∈ {"none", "ar+rates"}
      sign ∈ {"all"} or {"pos","neg"} if sign_split=True
      include_t0=True, two_sided=True, agg=winsor_mean (winsor=0.01)
    """
    idx = df.index
    ensure_dir(irf_dir)

    pctls = [0.85, 0.90, 0.95]
    resid_modes = ["none", "ar+rates"]
    signs = ["pos", "neg"] if sign_split else ["all"]

    rows: list[dict] = []

    for asset, cfg in ASSET_CFG.items():
        ycol, level_col, delta_col, flag_col = cfg["ret"], cfg["level"], cfg["delta"], cfg["flag"]
        if ycol not in df.columns:
            print(f"↪ Skipping {asset}: missing {ycol}")
            continue

        # Precompute return variants for residualization modes
        y_base = df[ycol].copy()
        y_resid = {
            "none": y_base,
            "ar+rates": residualize(y_base, df, ycol, "ar+rates"),
        }
        # winsorize both once
        for key in list(y_resid.keys()):
            y_resid[key] = winsorize(y_resid[key], 0.01)

        s_metric = shock_series_auto(df, level_col, delta_col, two_sided=True)

        for p in pctls:
            for sign in signs:
                mask = threshold_mask(s_metric, pctl=p, two_sided=True, sign=sign)
                shocks = first_of_run_with_cooldown(mask, idx, cooldown)
                n_total = int(mask.sum())

                for resid_mode in resid_modes:
                    y = y_resid[resid_mode]
                    mat = build_irf(idx, y, shocks, horizon, include_t0=True)
                    n_used = mat.shape[0]

                    if n_used == 0:
                        irf_avg = np.full(horizon, np.nan)
                        alpha = delta = r2_fit = half = np.nan
                    else:
                        irf_avg = np.nanmean(mat, axis=0)
                        alpha, delta, r2_fit = fit_decay(irf_avg)
                        half = (np.log(2) / delta) if (np.isfinite(delta) and delta > 0) else np.nan

                    # save IRF per config to irfs/
                    tag = f"p{int(round(p*100))}__resid_{resid_mode}__sign_{sign}"
                    irf_fp = irf_dir / f"irf_{asset.lower()}__{tag}.csv"
                    pd.DataFrame({"t": np.arange(0, horizon), "cum_ret": irf_avg}).to_csv(irf_fp, index=False)

                    print(f"{asset} [{tag}]: shocks(total/used)={n_total}/{n_used}; "
                          f"δ={delta if np.isfinite(delta) else np.nan:.4f}  "
                          f"α={alpha if np.isfinite(alpha) else np.nan:.4e}  "
                          f"R²_fit={r2_fit if np.isfinite(r2_fit) else np.nan:.3f}  "
                          f"half={half if np.isfinite(half) else np.nan:.2f}d")

                    rows.append({
                        "asset": asset, "pctl": p, "residualize": resid_mode, "sign": sign,
                        "alpha": alpha, "delta": delta, "half_life_days": half, "fit_r2": r2_fit,
                        "n_shocks_total": int(n_total), "n_shocks_used": int(n_used),
                        "horizon": int(horizon), "cooldown": int(cooldown),
                        "agg": "winsor_mean", "winsor": 0.01, "two_sided": True,
                        "include_t0": True, "use_flags": False,
                        "start": str(idx.min().date()), "end": str(idx.max().date()),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(base / PARAMS_ROBUST, index=False)
    print(f"✓ {PARAMS_ROBUST} written: {base / PARAMS_ROBUST}  shape={out.shape}")
    print(f"ℹ️ IRFs saved in: {irf_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str)
    ap.add_argument("--mode", choices=["single", "robust"], default="robust")
    # single-mode arguments (ignored in robust mode)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--pctl", type=float, default=0.90)
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--agg", choices=["mean", "winsor_mean", "median"], default="winsor_mean")
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--two_sided", action="store_true")
    ap.add_argument("--residualize", choices=["none", "ar", "rates", "ar+rates"], default="none")
    ap.add_argument("--use_flags", action="store_true", help="use *_shock flags from features instead of auto shocks")
    inc = ap.add_mutually_exclusive_group()
    inc.add_argument("--include_t0", dest="include_t0", action="store_true")
    inc.add_argument("--no-include_t0", dest="include_t0", action="store_false")
    ap.set_defaults(include_t0=True)
    # robust-only toggle
    ap.add_argument("--sign_split", action="store_true", help="in robust mode, also run positive/negative shocks separately")
    args = ap.parse_args()

    base = pick_base(args.base)
    feats_fp = base / FEATURES_NAME
    if not feats_fp.exists():
        sys.exit(f"Could not find {feats_fp}")

    # Always save IRFs to a tidy subfolder
    irf_dir = base / "irfs"
    ensure_dir(irf_dir)

    df = pd.read_csv(feats_fp, parse_dates=["date"], index_col="date").sort_index()
    cooldown = args.cooldown or args.horizon

    if args.mode == "single":
        run_single(df, irf_dir, base,
                   horizon=args.horizon,
                   pctl=args.pctl,
                   cooldown=cooldown,
                   agg=args.agg,
                   winsor=args.winsor,
                   two_sided=args.two_sided,
                   residualize_mode=args.residualize,
                   use_flags=args.use_flags,
                   include_t0=args.include_t0)
    else:
        # Robust grid (fixed settings as documented)
        run_robust(df, irf_dir, base,
                   horizon=args.horizon,
                   cooldown=cooldown,
                   sign_split=args.sign_split)


if __name__ == "__main__":
    main()

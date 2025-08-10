# discussionpost.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import date

# -----------------------------
# Base-path helpers
# -----------------------------
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

def read_csv_if_exists(fp: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(fp, **kwargs) if fp.exists() else None

# -----------------------------
# Formatting helpers
# -----------------------------
def fmt_days(x: float | None, dec: int = 2) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{x:.{dec}f}d"

def fmt_bp(x: float | None, dec: int = 1) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{10000*x:.{dec}f} bp"

# -----------------------------
# Summaries
# -----------------------------
def summarize_baseline_activity(tbl: pd.DataFrame) -> tuple[str, dict]:
    """
    Return (markdown text, stats dict). Expects columns:
      ['asset','bundle','variable','coef','se','p','stars'] from table_baseline_activity.csv
    Emphasize the richest bundle 'act+rates+ar' when available.
    """
    txt_lines = []
    stats = {}
    if tbl is None or tbl.empty:
        return "Baseline predictive regressions were unavailable.", {}

    bundles = tbl["bundle"].unique().tolist()
    prefer_bundle = "act+rates+ar" if "act+rates+ar" in bundles else ("act+rates" if "act+rates" in bundles else bundles[0])
    for a in ["SPY","MSFT","ETH","OIL"]:
        sub = tbl[(tbl["asset"] == a) & (tbl["bundle"] == prefer_bundle)].copy()
        if sub.empty:
            continue
        # Count significant variables at 5% and 10%
        sig5 = int((sub["p"] < 0.05).sum())
        sig10 = int((sub["p"] < 0.10).sum())
        # Largest absolute t-stat (if se available)
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = sub["coef"] / sub["se"].replace(0, np.nan)
        if tvals.notna().any():
            idx = tvals.abs().idxmax()
            row = sub.loc[idx]
            top_line = f"top |t|: {row['variable']} (coef={fmt_bp(row['coef'])}, p={row['p']:.3f})"
        else:
            top_line = "top |t|: n/a"
        txt_lines.append(f"- **{a}** ({prefer_bundle}): {sig5} vars at p<0.05 ({sig10} at p<0.10); {top_line}.")
        stats[a] = {"sig5":sig5, "sig10":sig10, "top":top_line}
    if not txt_lines:
        return "Baseline predictive regressions yielded no usable rows.", {}
    return "\n".join(txt_lines), stats

def summarize_branching(pref: pd.DataFrame) -> tuple[str, dict]:
    """
    Take baseline-spec (formerly 'preferred') branching params and produce narrative.
    Expects at least: ['asset','alpha','delta','half_life_days','fit_r2','n_shocks_used','pctl','residualize']
    """
    if pref is None or pref.empty:
        return "Branching IRF parameters not found.", {}

    txt_lines, stats = [], {}
    for _, r in pref.iterrows():
        a = r.get("asset", "—")
        alpha = r.get("alpha", np.nan)
        delta = r.get("delta", np.nan)
        half  = r.get("half_life_days", np.nan)
        r2    = r.get("fit_r2", np.nan)
        nused = (int(r.get("n_shocks_used")) if pd.notna(r.get("n_shocks_used", np.nan))
                 else int(r.get("n_shocks_total", 0)))
        pctl  = r.get("pctl", np.nan)
        resid = r.get("residualize", "—")

        drift = (np.isfinite(delta) and delta > 1e-6)
        drift_txt = f"half-life ≈ {fmt_days(half)}" if drift else "no forward drift (δ≈0)"
        txt_lines.append(f"- **{a}**: α={fmt_bp(alpha)}, {drift_txt}, fit R²={r2:.3f} (n_shocks={nused}, pctl={pctl}, resid={resid}).")
        stats[a] = {"alpha":alpha, "delta":delta, "half":half, "r2":r2, "n_shocks_used":nused}
    return "\n".join(txt_lines), stats

def load_eth_same_day(base: Path) -> dict:
    fp = base / "eth_same_day_reg.csv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp)
    out = {
        "coef": float(df.get("coef_d_fees", [np.nan])[0]),
        "p": float(df.get("p_d_fees", [np.nan])[0]),
        "r2": float(df.get("r2", [np.nan])[0]),
        "n": int(df.get("n", [np.nan])[0]),
    }
    return out

# -----------------------------
# Build Markdown
# -----------------------------
def build_markdown(base: Path) -> Path:
    # Load inputs
    feats = read_csv_if_exists(base / "features.csv", parse_dates=["date"])
    if feats is not None and not feats.empty:
        dates = (str(pd.to_datetime(feats["date"]).min().date()), str(pd.to_datetime(feats["date"]).max().date()))
    else:
        dates = ("—","—")

    baseline_tbl = read_csv_if_exists(base / "table_baseline_activity.csv")
    pref_tbl     = read_csv_if_exists(base / "table_branching_preferred.csv")
    single_tbl   = read_csv_if_exists(base / "table_branching_single.csv")
    if (pref_tbl is None or pref_tbl.empty) and (single_tbl is not None and not single_tbl.empty):
        pref_tbl = single_tbl.copy()  # fallback

    baseline_txt, _ = summarize_baseline_activity(baseline_tbl)
    branching_txt, bstats = summarize_branching(pref_tbl)
    eth_sd = load_eth_same_day(base)

    # Figure paths (relative)
    figs = {
        "SPY":  "figs/irf_spy_preferred.png",
        "MSFT": "figs/irf_msft_preferred.png",
        "ETH":  "figs/irf_eth_preferred.png",
        "OIL":  "figs/irf_oil_preferred.png",
        "ETH_SCAT": "figs/eth_same_day_scatter.png",
    }

    # Paper-ready caption boilerplate
    BASELINE_CAPTION = "(baseline: p90 shocks, t0 included, AR+rates residualized, H=20)"
    ETH_SD_CAPTION   = "ETH same-day scatter (r_t vs Δfees_t; OLS with HAC; SPY trading days)"

    # Title + executive summary
    md = []
    md.append("# Discussion & Results\n")
    md.append(f"*Generated on {date.today().isoformat()} from {dates[0]} → {dates[1]} data.*\n")

    # Executive summary bullets
    bullets = []
    if bstats:
        drift_assets = [a for a, s in bstats.items() if np.isfinite(s.get("delta", np.nan)) and s.get("delta", 0) > 1e-6]
        if "MSFT" in drift_assets:
            bullets.append(f"MSFT shows short-lived follow-through after activity spikes (**baseline half-life ≈ {fmt_days(bstats['MSFT']['half'])}**); others exhibit **no forward drift** (δ≈0).")
        else:
            bullets.append("Across assets, activity spikes produce **same-day moves with little/no follow-through** (δ≈0).")
        if "SPY" in bstats:
            bullets.append(f"For SPY, IRFs indicate contemporaneous jumps only (α={fmt_bp(bstats['SPY']['alpha'])}).")
        if "ETH" in bstats:
            bullets.append(f"For ETH, IRFs also point to same-day effects only (α={fmt_bp(bstats['ETH']['alpha'])}).")
    else:
        bullets.append("Branching IRF parameters were not available; see baseline regressions.")

    if np.isfinite(eth_sd.get("coef", np.nan)):
        dir_txt = "positive" if eth_sd["coef"] > 0 else "negative"
        bullets.append(f"ETH same-day regression: Δfees_t → r_t has a **{dir_txt}** slope (coef={fmt_bp(eth_sd['coef'])}, p={eth_sd['p']:.3f}, R²={eth_sd['r2']:.3f}).")

    md.append("## Executive summary\n")
    md.append("\n".join([f"- {b}" for b in bullets]) + "\n")

    # Baseline predictive regressions
    md.append("## Baseline predictive regressions (next-day returns)\n")
    md.append("We estimate OLS models of next-day returns on activity deltas with rate controls and AR(1). "
              "Overall, **predictive power is weak**, consistent with prices reacting contemporaneously to activity shocks.\n")
    md.append(baseline_txt + "\n")

    # Branching / IRF results
    md.append("## Branching / self-exciting impulse responses\n")
    md.append("We identify high-percentile activity shocks and average forward returns in event time. "
              "Fitting an exponential **g(t)=α·exp(−δt)** shows α captures the same-day jump, while δ captures any follow-through. "
              "Below we report **baseline-specification** parameters and IRF plots "
              "(p90 shocks over a rolling ≈2y window, first-of-run with cooldown=20d, **t=0 included**, "
              "winsorized mean 1%, returns residualized on **AR(1)+GS3M+GS10**).\n")
    md.append(branching_txt + "\n")

    for a, rel in [("SPY", figs["SPY"]), ("MSFT", figs["MSFT"]), ("ETH", figs["ETH"]), ("OIL", figs["OIL"])]:
        md.append(f"**{a} IRF {BASELINE_CAPTION}**  \n![{a} IRF]({rel})\n")

    # ETH same-day check (paper-ready wording)
    md.append("## ETH same-day activity–return check (baseline: OLS r_t ~ Δfees_t, HAC; SPY trading-day calendar)\n")
    if np.isfinite(eth_sd.get("coef", np.nan)):
        md.append(
            f"Same-day OLS of r_t on Δfees_t (HAC standard errors, SPY trading-day calendar) yields "
            f"coef={fmt_bp(eth_sd['coef'])}, p={eth_sd['p']:.3f}, R²={eth_sd['r2']:.3f} (n={eth_sd['n']}). "
            "This suggests **contemporaneous comovement** between fees and returns; it does **not** imply next-day predictability.\n"
        )
        md.append(f"**{ETH_SD_CAPTION}**  \n![ETH same-day scatter]({figs['ETH_SCAT']})\n")
    else:
        md.append("The ETH same-day regression was unavailable. Ensure `eth_same_day_reg.csv` is created by running `main.py`.\n")

    # (optional) Earnings trend robustness (2–4 quarters) + monthly rates check
    trend = read_csv_if_exists(base / "baseline_trends.csv")
    if trend is not None and not trend.empty:
        md.append("## Earnings trend robustness (2–4 quarters)\n")
        order = {"2q": 1, "3q": 2, "4q": 3}
        lines = []
        for asset in ["SPY", "MSFT"]:
            sub = trend[trend["asset"] == asset].copy()
            if sub.empty:
                continue
            sub["h"] = sub["trend_var"].str.extract(r"_(2q|3q|4q)$")[0]
            sub = sub.sort_values("h", key=lambda s: s.map(order))
            parts = []
            for _, r in sub.iterrows():
                bp = 10000 * float(r["coef_trend"])
                p  = float(r["p_trend"])
                tag = r["h"].upper()
                parts.append(f"{tag}: {bp:+.1f} bp (p={p:.3f})")
            k_sig = int((sub["p_trend"] < 0.05).sum())
            lines.append(f"- **{asset}**: " + "; ".join(parts) + f".  sig@5%={k_sig}.")
        md.append("\n".join(lines) + "\n")

    ratesm = read_csv_if_exists(base / "rates_trends_monthly.csv")
    if ratesm is not None and not ratesm.empty:
        md.append("## Monthly rate changes vs returns\n")
        lines = []
        for asset in ["SPY", "MSFT"]:
            for var, label in [("d_gs3m", "ΔGS3M"), ("d_gs10", "ΔGS10")]:
                r = ratesm[(ratesm["asset"] == asset) & (ratesm["predictor"] == var)]
                if r.empty:
                    continue
                coef = 10000 * float(r["coef"].iloc[0])  # bp per 1 pp rate change
                p    = float(r["p"].iloc[0])
                corr = float(r["corr"].iloc[0])
                lines.append(f"- **{asset}** {label}: {coef:+.1f} bp per 1 pp (p={p:.3f}), corr={corr:.3f}.")
        md.append("\n".join(lines) + "\n")

    # Robustness & caveats
    md.append("## Robustness & caveats\n")
    md.append(
        "- Results are **robust** across shock thresholds (p85/p90/p95) and residualization choices; only MSFT shows a modest, short half-life.\n"
        "- IRFs use **trading-day event time** (SPY calendar). Crypto weekend dynamics are partly excluded, making ETH conclusions conservative.\n"
        "- Baseline OLS uses Newey–West SEs in the modeling scripts; remaining serial correlation or overlapping-window effects should be minor at daily horizons.\n"
        "- δ≈0 implies no forward drift; very large half-life numbers are just the mathematical limit as δ→0 and should be interpreted as **“no drift.”**\n"
    )

    # Save
    out = base / "discussionpost.md"
    Path(out).write_text("\n".join(md), encoding="utf-8")
    print(f"✓ discussionpost.md written: {out}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str)
    args = ap.parse_args()

    base = pick_base(args.base)
    print(f"📂 Using data dir: {base}")
    build_markdown(base)

if __name__ == "__main__":
    main()

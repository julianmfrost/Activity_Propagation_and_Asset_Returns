# clean.merge.py

from __future__ import annotations
import argparse
from functools import reduce
from pathlib import Path
import sys
import warnings
import pandas as pd


# ---------------------------------------------------------------------
# Defaults & file config
# ---------------------------------------------------------------------
DEFAULT_BASE_CANDIDATES = [
    Path("/Users/julianfrost/Documents/PSP_Analysis/data"),
    Path("/mnt/data"),
    Path.cwd() / "data",
]
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2026-03-31"

PRICE_FILES = ["spy_price", "msft_price", "eth_price", "oil_price"]
# name -> (date_column, freq_flag)
# freq_flag: 'Q' (quarterly), 'M' (monthly), None (already daily)
ACTIVITY_FILES = {
    "spy_earnings": ("quarter end", "Q"),
    "msft_earnings": ("quarter end", "Q"),
    "eth_fees": ("Date", None),
    "oil_consumption": ("Date", "M"),
    "GS3M": ("observation_date", None),   
    "GS10": ("observation_date", None),   
}



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


def _parse_dates(series: pd.Series) -> pd.DatetimeIndex:
    """
    Try strict formats first to avoid warnings; fall back to dateutil on mixed data.
    """
    # Attempt ISO first
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if parsed.notna().mean() >= 0.99:
        return parsed

    # Attempt US-style MDY
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    if parsed.notna().mean() >= 0.99:
        return parsed

    # Fallback to flexible parsing (silence the pandas warning noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce")


def load_csv(base: Path, name: str, *, date_col: str = "Date", freq: str | None = None) -> pd.DataFrame:
    """
    Standardize a single CSV:
      - Renames second column to `name`
      - Parses date with strict formats (ISO, then MDY), fallback to dateutil
      - Coerces numeric (strips $, commas, etc.)
      - Quarterly/Monthly series are upsampled to business days via ffill
    """
    fp = base / f"{name}.csv"
    if not fp.exists():
        sys.exit(f"Missing expected file: {fp}")

    df = pd.read_csv(fp)
    if df.shape[1] < 2:
        sys.exit(f"{fp} must have at least two columns (date, value).")

    value_col = df.columns[1]
    df = df.rename(columns={date_col: "date", value_col: name})
    if "date" not in df.columns:
        sys.exit(f"{fp} missing expected date column '{date_col}'.")

    # Parse dates (robust), drop bad/missing, de-dup, sort
    parsed = _parse_dates(df["date"])
    df = df.assign(date=parsed).dropna(subset=["date"]).drop_duplicates(subset=["date"])
    df = df.set_index("date").sort_index()

    # Robust numeric coercion
    s = (
        df[name]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None})
    )
    df[name] = pd.to_numeric(s, errors="coerce")

    # Quarterly / Monthly → business-day grid with ffill
    if freq in {"Q", "M"}:
        df = df.resample("B").ffill()

    return df[[name]]


def join_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        sys.exit("No data frames to merge.")
    return reduce(lambda l, r: l.join(r, how="outer"), frames)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, help="Directory containing the CSV files")
    ap.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    args = ap.parse_args()

    base = pick_base(args.base)
    print(f"📂 Using data directory: {base}")

    # Load all series
    frames: list[pd.DataFrame] = []

    # Prices (already daily; some skip weekends/holidays)
    for name in PRICE_FILES:
        df = load_csv(base, name, date_col="Date", freq=None)
        frames.append(df)
        print(f"  ✓ Loaded {name}: {df.index.min().date()} → {df.index.max().date()} (n={len(df)})")

    # Activity / fundamentals (mixed frequency)
    for name, (dcol, fflag) in ACTIVITY_FILES.items():
        df = load_csv(base, name, date_col=dcol, freq=fflag)
        frames.append(df)
        print(f"  ✓ Loaded {name}: {df.index.min().date()} → {df.index.max().date()} (n={len(df)})")

    # Merge → restrict window (no ffill yet)
    master = join_frames(frames)
    master = master.loc[args.start:args.end].sort_index()

    # Keep SPY trading days FIRST (prevents weekend forward-fill issues)
    if "spy_price" not in master.columns:
        sys.exit("'spy_price' column missing after merge; cannot filter to trading days.")
    before = len(master)
    master = master.loc[master["spy_price"].notna()].copy()
    after = len(master)

    # Now forward-fill on the trading-day index only
    master = master.ffill()

    # Save
    out_path = base / "master.csv"
    master.to_csv(out_path, index=True)  # index name 'date' is preserved
    print(
        f"✔ master.csv written: {out_path}\n"
        f"   rows={after} (dropped {before - after} non-trading days), "
        f"cols={master.shape[1]}, "
        f"range={master.index.min().date()} → {master.index.max().date()}"
    )


if __name__ == "__main__":
    main()

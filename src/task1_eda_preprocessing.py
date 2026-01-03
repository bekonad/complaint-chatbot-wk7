# src/task1_eda_preprocessing.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

# ── CONFIG ───────────────────────────────────────────────────────
RAW_PATH   = Path("../data/raw/complaints.csv")
OUT_DIR    = Path("../data/processed")
OUT_CSV    = OUT_DIR / "filtered_complaints.csv"
FIGS_DIR   = OUT_DIR / "figures"
NROWS      = 300_000

FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RELEVANT = [
    "Credit card",
    "Credit card or prepaid card",
    "Prepaid card",
    "Checking or savings account",
    "Payday loan, title loan, personal loan, or advance loan",
    "Money transfer, virtual currency, or money service"
]

def save_fig(name):
    path = FIGS_DIR / f"{name}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.name}")

def main():
    print(f"Loading {NROWS:,} rows...")
    try:
        df = pd.read_csv(RAW_PATH, nrows=NROWS, low_memory=False)
    except FileNotFoundError:
        print(f"File not found: {RAW_PATH}")
        sys.exit(1)
    except Exception as e:
        print("Load error:", str(e))
        sys.exit(1)

    print("Shape:", df.shape)

    # Quick overview
    print("\nTop products:\n", df["Product"].value_counts().head(10))
    print("\nHas narrative: {:.1%}".format(
        df["Consumer complaint narrative"].notna().mean()))

    # Plots
    counts = df["Product"].value_counts()
    plt.figure(figsize=(8,5))
    sns.barplot(x=counts.head(10).values, y=counts.head(10).index)
    plt.title("Top 10 Products")
    save_fig("top_products")

    has_text = df[df["Consumer complaint narrative"].notna()].copy()
    has_text["words"] = has_text["Consumer complaint narrative"].str.split().str.len()
    print("\nWord stats:\n", has_text["words"].describe().round(1))

    plt.figure(figsize=(8,4))
    sns.histplot(has_text["words"], bins=50)
    plt.title("Narrative length")
    plt.xlim(0, 600)
    save_fig("narrative_length")

    # Filter + clean
    df_f = df[
        df["Product"].isin(RELEVANT) &
        df["Consumer complaint narrative"].notna() &
        df["Consumer complaint narrative"].str.strip().ne("")
    ].copy()

    print("\nAfter filter:", df_f.shape)

    def clean(t):
        t = str(t).lower().strip()
        t = re.sub(r'[^a-z0-9\s\.\,\!\?\'\-]', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    df_f["cleaned"] = df_f["Consumer complaint narrative"].apply(clean)
    df_clean = df_f[df_f["cleaned"].str.strip().ne("")]

    print("After clean:", df_clean.shape)

    df_clean.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")
    print(f"Rows: {len(df_clean):,}")

if __name__ == "__main__":
    main()
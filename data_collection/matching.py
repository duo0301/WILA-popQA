"""
Hard distribution matching (Option A: per-bin minimum across languages)
on the given JSON files: Arabic.json, Chinese.json, Russian.json.

Produces:
- matched_ar_qids.csv, matched_zh_qids.csv, matched_ru_qids.csv
- matched_ar_zh_ru_qids.csv (combined)
- hard_match_target_histogram.csv (bin counts + target)
- hard_matched_ar_zh_ru_sitelinks_summary.csv (summary stats before/after)
- hard_matched_ar_zh_ru_overlap_plots.pdf (density + ECDF)

Assumes JSON structure:
{ "Q123": {"sitelinks": 16, ...}, ... }
"""

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------------ Paths ------------------
ARABIC_PATH  = "data/dataset_v2/entities_data_complete/Arabic.json"
CHINESE_PATH = "data/dataset_v2/entities_data_complete/Chinese.json"
ENGLISH_PATH = "data/dataset_v2/entities_data_complete/English.json"
FRENCH_PATH = "data/dataset_v2/entities_data_complete/French.json"
GERMAN_PATH = "data/dataset_v2/entities_data_complete/German.json"
HINDI_PATH = "data/dataset_v2/entities_data_complete/Hindi.json"
ITALIAN_PATH = "data/dataset_v2/entities_data_complete/Italian.json"
POLISH_PATH = "data/dataset_v2/entities_data_complete/Polish.json"
RUSSIAN_PATH = "data/dataset_v2/entities_data_complete/Russian.json"

OUT_MATCHED_COMBINED = "data/dataset_v2/entities_data_matched/matched_qids.csv"

OUT_HIST = "data/dataset_v2/entities_data_matched/hard_match_histogram.csv"
OUT_SUMMARY = "data/dataset_v2/entities_data_matched/hard_matched_sitelinks_summary.csv"
OUT_PDF = "data/dataset_v2/entities_data_matched/hard_matched_overlap_plots.pdf"

BIN_WIDTH = 5
RANDOM_SEED = 42


# ------------------ Helpers ------------------
def load_df(json_path: str) -> pd.DataFrame:
    """Load qid + sitelinks from a language JSON file with sanity checks."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    rows = []
    for qid, obj in d.items():
        if not isinstance(obj, dict):
            continue
        s = obj.get("sitelinks", None)
        if isinstance(s, (int, float)) and not (isinstance(s, float) and math.isnan(s)):
            s = int(s)
            if s < 0:
                raise ValueError(f"Negative sitelinks for {qid} in {json_path}")
            rows.append((qid, s))

    df = pd.DataFrame(rows, columns=["qid", "sitelinks"])
    if df.empty:
        raise ValueError(f"No valid sitelinks found in {json_path}")
    if df["qid"].duplicated().any():
        raise ValueError(f"Duplicate QIDs found in {json_path}")
    return df


def make_bin_edges(global_min: int, global_max: int, width: int) -> list[int]:
    """Create [edge_i, edge_{i+1}) bins that fully cover [global_min, global_max]."""
    start = (global_min // width) * width
    end = ((global_max // width) + 2) * width
    edges = list(range(start, end + 1, width))
    if edges[0] > global_min or edges[-1] < global_max:
        raise ValueError("Bin edges do not cover the full sitelinks range.")
    return edges


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, p90=np.nan, p95=np.nan, p99=np.nan)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def ecdf(arr: np.ndarray):
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def hard_match_by_sitelinks(dfs_by_lang: dict, bin_edges: list[int], seed: int = 0):
    """
    Option A hard matching:
      - bin sitelinks with common edges
      - target count per bin = min across languages
      - sample exactly target per bin for each language
    """
    rng = np.random.default_rng(seed)

    # Bin each language
    binned = {}
    for lang, df in dfs_by_lang.items():
        df = df.copy()

        if df["sitelinks"].isna().any():
            raise ValueError(f"{lang}: NaN sitelinks present")
        if (df["sitelinks"] < 0).any():
            raise ValueError(f"{lang}: negative sitelinks present")

        df["bin"] = pd.cut(df["sitelinks"], bins=bin_edges, right=False, include_lowest=True)
        df = df.dropna(subset=["bin"])  # drop out-of-range just in case
        binned[lang] = df

    # Counts per bin per language
    counts = pd.DataFrame({
        lang: df["bin"].value_counts().sort_index()
        for lang, df in binned.items()
    }).fillna(0).astype(int)

    # Target is min across languages (feasible by construction)
    target = counts.min(axis=1)
    target = target[target > 0]  # drop bins where any language has 0

    # Sample within each bin
    matched = {}
    for lang, df in binned.items():
        pieces = []
        for bin_label, tb in target.items():
            sub = df[df["bin"] == bin_label]
            if len(sub) < int(tb):
                # Should never happen if target = min counts
                raise RuntimeError(f"Feasibility violated for {lang} in bin {bin_label}")
            idx = rng.choice(sub.index.to_numpy(), size=int(tb), replace=False)
            pieces.append(sub.loc[idx])

        matched_df = pd.concat(pieces, ignore_index=True)
        matched[lang] = matched_df

    # Sanity checks: histograms match target EXACTLY
    for lang, df in matched.items():
        got = (
            df["bin"].value_counts().sort_index()
            .reindex(target.index).fillna(0).astype(int)
        )
        if not (got.values == target.values).all():
            raise AssertionError(f"{lang}: matched histogram != target histogram")

    # Every language should have the same total selected size
    totals = {lang: len(df) for lang, df in matched.items()}
    if len(set(totals.values())) != 1:
        raise AssertionError(f"Totals differ across languages: {totals}")

    return matched, target, counts


# ------------------ Run ------------------
ar = load_df(ARABIC_PATH)
zh = load_df(CHINESE_PATH)
ru = load_df(RUSSIAN_PATH)

arabic = load_df(ARABIC_PATH)
chinese = load_df(CHINESE_PATH)
english = load_df(ENGLISH_PATH)
french = load_df(FRENCH_PATH)
german = load_df(GERMAN_PATH)
hindi = load_df(HINDI_PATH)
italian = load_df(ITALIAN_PATH)
polish = load_df(POLISH_PATH)
russian = load_df(RUSSIAN_PATH)

dfs = {
    "Arabic": arabic,
    "Chinese": chinese,
    "English": english,
    "French": french, 
    "German": german,
    "Hindi": hindi,
    "Italian": italian,
    "Polish": polish,
    "Russian": russian,
}

global_min = int(min(df["sitelinks"].min() for df in dfs.values()))
global_max = int(max(df["sitelinks"].max() for df in dfs.values()))
bin_edges = make_bin_edges(global_min, global_max, BIN_WIDTH)

matched, target, counts = hard_match_by_sitelinks(dfs, bin_edges, seed=RANDOM_SEED)

# ------------------ Save matched QIDs ------------------

combined = pd.concat(
    [matched[lang][["qid", "sitelinks"]].assign(lang=lang) for lang in dfs.keys()],
    ignore_index=True
)
combined.to_csv(OUT_MATCHED_COMBINED, index=False)

# ------------------ Save target histogram ------------------
hist_table = pd.DataFrame({
    lang: matched[lang]["bin"].value_counts().sort_index().reindex(target.index).fillna(0).astype(int)
    for lang in dfs.keys()
})
hist_table["target"] = target.astype(int)
hist_table.reset_index().rename(columns={"bin": "bin_interval"}).to_csv(OUT_HIST, index=False)

# ------------------ Summary before/after ------------------
rows = []
for lang, df in dfs.items():
    rows.append({"lang": lang, "stage": "original", **summarize(df["sitelinks"].to_numpy())})
for lang, df in matched.items():
    rows.append({"lang": lang, "stage": "matched", **summarize(df["sitelinks"].to_numpy())})

summary_df = pd.DataFrame(rows).sort_values(["stage", "lang"])
summary_df.to_csv(OUT_SUMMARY, index=False)

# ------------------ Plots (matched only) ------------------
with PdfPages(OUT_PDF) as pdf:
    # Density histograms
    plt.figure(figsize=(10, 4))
    for lang in dfs.keys():
        plt.hist(
            matched[lang]["sitelinks"],
            bins=bin_edges,
            density=True,
            alpha=0.5,
            label=f"{lang} (n={len(matched[lang])})"
        )
    plt.title("Hard-matched sitelinks distributions (density histograms)")
    plt.xlabel("Sitelinks")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ECDFs
    plt.figure(figsize=(10, 4))
    for lang in dfs.keys():
        x, y = ecdf(matched[lang]["sitelinks"].to_numpy())
        plt.step(x, y, where="post", label=f"{lang} ECDF")
    plt.title("Hard-matched sitelinks distributions (ECDF)")
    plt.xlabel("Sitelinks")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("DONE. Wrote:")
print(" -", OUT_MATCHED_COMBINED)
print(" -", OUT_HIST)
print(" -", OUT_SUMMARY)
print(" -", OUT_PDF)
print("\nMatched N per language:", {k: len(v) for k, v in matched.items()})

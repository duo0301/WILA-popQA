import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Inputs ----------
ARABIC_PATH  = "data/dataset_v2/entities_data_complete/Arabic.json"
CHINESE_PATH = "data/dataset_v2/entities_data_complete/Chinese.json"
ENGLISH_PATH = "data/dataset_v2/entities_data_complete/English.json"
FRENCH_PATH = "data/dataset_v2/entities_data_complete/French.json"
GERMAN_PATH = "data/dataset_v2/entities_data_complete/German.json"
HINDI_PATH = "data/dataset_v2/entities_data_complete/Hindi.json"
ITALIAN_PATH = "data/dataset_v2/entities_data_complete/Italian.json"
POLISH_PATH = "data/dataset_v2/entities_data_complete/Polish.json"
RUSSIAN_PATH = "data/dataset_v2/entities_data_complete/Russian.json"


OUT_PDF = "data/dataset_v2/statistics/complete/sitelinks_overlap.pdf"
OUT_SUMMARY_CSV = "data/dataset_v2/statistics/complete/sitelinks_summary.csv"
OUT_KS_CSV = "data/dataset_v2/statistics/complete/pairwise_ks_two_sided.csv"

BIN_WIDTH = 5  # keep consistent with earlier plots

# ---------- Helpers ----------
def load_sitelinks(json_path: str) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sitelinks = []
    for _, obj in data.items():
        if not isinstance(obj, dict):
            continue
        s = obj.get("sitelinks", None)
        if isinstance(s, (int, float)) and not (isinstance(s, float) and math.isnan(s)):
            sitelinks.append(int(s))

    arr = np.array(sitelinks, dtype=int)

    # Sanity checks to avoid silent errors
    if arr.size == 0:
        raise ValueError(f"No valid sitelinks found in: {json_path}")
    if np.any(arr < 0):
        raise ValueError(f"Negative sitelinks found in: {json_path}")
    return arr

def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
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

def pairwise_ks_two_sided(datasets: dict) -> pd.DataFrame:
    """
    Returns KS D-stat + p-value for all unordered pairs.
    Uses SciPy if available; otherwise computes D only (p_value=None).
    """
    names = list(datasets.keys())
    rows = []

    try:
        from scipy.stats import ks_2samp
        scipy_ok = True
    except Exception:
        scipy_ok = False

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = datasets[a_name], datasets[b_name]

            if scipy_ok:
                stat, p = ks_2samp(a, b, alternative="two-sided", method="auto")
                rows.append({"A": a_name, "B": b_name, "KS_D": float(stat), "p_value": float(p)})
            else:
                # Manual KS statistic (no p-value)
                a_sorted = np.sort(a)
                b_sorted = np.sort(b)
                all_vals = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
                a_cdf = np.searchsorted(a_sorted, all_vals, side="right") / a_sorted.size
                b_cdf = np.searchsorted(b_sorted, all_vals, side="right") / b_sorted.size
                d = float(np.max(np.abs(a_cdf - b_cdf)))
                rows.append({"A": a_name, "B": b_name, "KS_D": d, "p_value": None})

    return pd.DataFrame(rows)

# ---------- Load data ----------
arabic = load_sitelinks(ARABIC_PATH)
chinese = load_sitelinks(CHINESE_PATH)
english = load_sitelinks(ENGLISH_PATH)
french = load_sitelinks(FRENCH_PATH)
german = load_sitelinks(GERMAN_PATH)
hindi = load_sitelinks(HINDI_PATH)
italian = load_sitelinks(ITALIAN_PATH)
polish = load_sitelinks(POLISH_PATH)
russian = load_sitelinks(RUSSIAN_PATH)

datasets = {
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

# Extra safety checks (double-check ranges)
global_min = min(int(v.min()) for v in datasets.values())
global_max = max(int(v.max()) for v in datasets.values())
if global_min != 9:
    raise ValueError(f"Unexpected global min sitelinks: {global_min} (expected 9 in your runs)")
if global_max < global_min:
    raise ValueError("Invalid global range (max < min)")

# ---------- Summary CSV ----------
summary_df = pd.DataFrame([{"dataset": name, **summarize(arr)} for name, arr in datasets.items()])
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

# ---------- Pairwise KS (two-sided) CSV ----------
ks_df = pairwise_ks_two_sided(datasets)
ks_df.to_csv(OUT_KS_CSV, index=False)

# ---------- Plots PDF (overlap histograms + ECDF) ----------
# Bin edges chosen to fully cover [global_min, global_max]
start = (global_min // BIN_WIDTH) * BIN_WIDTH
end = ((global_max // BIN_WIDTH) + 2) * BIN_WIDTH  # +2 ensures last bin edge >= max
edges = np.arange(start, end, BIN_WIDTH)
if edges[0] > global_min or edges[-1] < global_max:
    raise ValueError("Histogram bin edges do not cover full data range")

with PdfPages(OUT_PDF) as pdf:
    # 1) Overlapped histogram (counts)
    plt.figure(figsize=(10, 4))
    for name, arr in datasets.items():
        plt.hist(arr, bins=edges, alpha=0.45, label=f"{name} (n={arr.size})")
    plt.title("Sitelinks distributions (overlapped histograms)")
    plt.xlabel("Sitelinks")
    plt.ylabel("Number of entities")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2) Overlapped histogram (density)
    plt.figure(figsize=(10, 4))
    for name, arr in datasets.items():
        plt.hist(arr, bins=edges, density=True, alpha=0.45, label=f"{name} (density)")
    plt.title("Sitelinks distributions (overlapped density histograms)")
    plt.xlabel("Sitelinks")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 3) ECDF overlap
    plt.figure(figsize=(10, 4))
    for name, arr in datasets.items():
        x, y = ecdf(arr)
        plt.step(x, y, where="post", label=f"{name} ECDF")
    plt.title("Sitelinks distributions (ECDF overlap)")
    plt.xlabel("Sitelinks")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Wrote:")
print("  PDF:", OUT_PDF)
print("  Summary CSV:", OUT_SUMMARY_CSV)
print("  Pairwise KS CSV:", OUT_KS_CSV)
print("\nSummary head:")
print(summary_df)
print("\nPairwise KS (two-sided):")
print(ks_df)
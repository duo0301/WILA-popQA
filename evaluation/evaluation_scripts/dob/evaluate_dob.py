"""
evaluate_dob.py — Evaluate date-of-birth predictions from LLMs.

Pure string matching — no API calls needed.

Scoring scheme (eval_score column):
    exact_match:            1.0   full Y-M-D match
    swap_match:             1.0   full match with MM/DD transposed
    year_month_match:       0.7   year + month correct, day wrong
    year_month_swap_match:  0.7   year + month correct after MM/DD swap
    year_match:             0.3   year only correct  (eval = FP)
    no_match:               0.0
    empty (FN):             0.0

Usage:
    python evaluate_dob.py --input raw.csv --output results.csv
    python evaluate_dob.py --batch-dir model/raw/ --output-dir model/eval/
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ─── Scoring ──────────────────────────────────────────────────────────────

SCORES: dict[str, float] = {
    "exact_match":           1.0,
    "swap_match":            1.0,
    "year_month_match":      0.7,
    "year_month_swap_match": 0.7,
    "year_match":            0.5,
    "no_match":              0.0,
    "empty":                 0.0,
}


# ─── Parsing ──────────────────────────────────────────────────────────────

def parse_date(s: str) -> tuple[str, str, str] | None:
    s = s.strip()
    s = s.split("T")[0]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


# ─── Matching ─────────────────────────────────────────────────────────────

def evaluate_dob(gt_raw: str, output_raw: str) -> tuple[str, str, float]:
    """
    Compare GT and LLM output dates.

    Returns (eval, eval_type, eval_score):
        eval       = TP | FP | FN
        eval_type  = exact_match | swap_match | year_month_match |
                     year_month_swap_match | year_match | no_match | empty
        eval_score = float in [0.0, 1.0]
    """
    output_raw = output_raw.strip()

    # FN: empty or NONE output
    if not output_raw or output_raw.upper() == "NONE":
        return "FN", "empty", SCORES["empty"]

    gt = parse_date(gt_raw)
    out = parse_date(output_raw)

    if gt is None:
        return "FP", "no_match", SCORES["no_match"]

    if out is None:
        # Try to extract a date from longer text
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", output_raw)
        if m:
            out = m.group(1), m.group(2), m.group(3)
        else:
            return "FP", "no_match", SCORES["no_match"]

    y_gt, m_gt, d_gt = gt
    y_out, m_out, d_out = out

    # Exact match
    if (y_gt, m_gt, d_gt) == (y_out, m_out, d_out):
        return "TP", "exact_match", SCORES["exact_match"]

    # Swap MM and DD in output
    if int(d_out) <= 12 and int(m_out) <= 31:
        if (y_gt, m_gt, d_gt) == (y_out, d_out, m_out):
            return "TP", "swap_match", SCORES["swap_match"]

    # Year + month match
    if y_gt == y_out and m_gt == m_out:
        return "TP", "year_month_match", SCORES["year_month_match"]

    # Year + month match after swap
    if int(d_out) <= 12 and y_gt == y_out and m_gt == d_out:
        return "TP", "year_month_swap_match", SCORES["year_month_swap_match"]

    # Year only match
    if y_gt == y_out:
        return "FP", "year_match", SCORES["year_match"]

    return "FP", "no_match", SCORES["no_match"]


# ─── Main ─────────────────────────────────────────────────────────────────

def safe_print(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)


def evaluate_file(input_path: str, output_path: str):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    safe_print(f"Loaded {len(rows)} rows from {input_path}")

    # Check if already evaluated (requires eval_score to be present too)
    if "eval" in fieldnames and "eval_type" in fieldnames and "eval_score" in fieldnames:
        safe_print(f"  Already evaluated — skipping.")
        return

    out_fieldnames = fieldnames + ["eval", "eval_type", "eval_score"]

    for row in rows:
        ev, ev_type, ev_score = evaluate_dob(row["gt"], row["llm_output"])
        row["eval"] = ev
        row["eval_type"] = ev_type
        row["eval_score"] = ev_score

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    eval_counts = Counter(r["eval"] for r in rows)
    type_counts = Counter(r["eval_type"] for r in rows if r["eval_type"])
    tp = eval_counts.get("TP", 0)
    fp = eval_counts.get("FP", 0)
    fn = eval_counts.get("FN", 0)
    total = len(rows)

    weighted_sum = sum(float(r["eval_score"]) for r in rows)
    weighted_avg = weighted_sum / total if total else 0.0

    safe_print(f"\n=== Summary for {os.path.basename(input_path)} ===")
    safe_print(f"  TP: {tp}")
    safe_print(f"  FP: {fp}")
    safe_print(f"  FN: {fn}")
    safe_print(f"  Weighted score (avg eval_score): {weighted_avg:.4f}")
    safe_print(f"  TP/FP breakdown:")
    for mt, cnt in type_counts.most_common():
        score = SCORES.get(mt, 0.0)
        safe_print(f"    {mt:28s} {cnt:5d}  (score={score})")
    safe_print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DOB predictions.")
    parser.add_argument("--input",      help="Single input CSV file")
    parser.add_argument("--output",     help="Output CSV file (single-file mode)")
    parser.add_argument("--batch-dir",  help="Directory of input CSVs")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    args = parser.parse_args()

    if args.batch_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --batch-dir")
        batch_dir = Path(args.batch_dir)
        output_dir = Path(args.output_dir)
        csvs = sorted(batch_dir.glob("property_dob_*.csv"))
        if not csvs:
            csvs = sorted(batch_dir.glob("*.csv"))
        safe_print(f"Found {len(csvs)} CSV files in {batch_dir}")
        for csv_path in csvs:
            out_path = output_dir / csv_path.name
            safe_print(f"\n{'='*60}")
            safe_print(f"Processing: {csv_path.name}")
            safe_print(f"{'='*60}")
            evaluate_file(str(csv_path), str(out_path))
    elif args.input:
        if not args.output:
            parser.error("--output is required with --input")
        evaluate_file(args.input, args.output)
    else:
        parser.error("Either --input or --batch-dir is required")


if __name__ == "__main__":
    main()

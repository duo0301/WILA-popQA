"""
evaluate_dob.py

Usage:
    python evaluate_dob.py --input scored.csv --output results_dob.csv

--input  : path to the CSV (columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
--output : path for the scored output CSV
"""

import argparse
import csv
import re


# ─── Parsing ──────────────────────────────────────────────────────────────

def parse_date(s: str) -> tuple[str, str, str] | None:
    """
    Extract YYYY, MM, DD from a string.
    Handles both ISO with timestamp (2024-01-03T00:00:00Z) and plain (2024-01-03).
    Returns None if no valid date pattern found.
    """
    s = s.strip()
    # Strip Wikidata timestamp suffix
    s = s.split("T")[0]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


# ─── Matching ─────────────────────────────────────────────────────────────

def evaluate_dob(gt_raw: str, output_raw: str) -> tuple[str, float]:
    """
    Compare GT and LLM output dates.

    Returns (match_type, score):
        exact_match   1.0   — YYYY-MM-DD matches directly
        swap_match    1.0   — match after swapping MM and DD in output
        no_match      0.0   — no match under either interpretation
    """
    gt = parse_date(gt_raw)
    out = parse_date(output_raw)

    if gt is None or out is None:
        return "no_match", 0.0

    y_gt, m_gt, d_gt = gt
    y_out, m_out, d_out = out

    # Exact match
    if (y_gt, m_gt, d_gt) == (y_out, m_out, d_out):
        return "exact_match", 1.0

    # Swap MM and DD in output — only valid if swapped values are plausible
    # (swapped month ≤ 12, swapped day ≤ 31)
    if int(d_out) <= 12 and int(m_out) <= 31:
        if (y_gt, m_gt, d_gt) == (y_out, d_out, m_out):
            return "swap_match", 1.0

    # Year + month match
    if y_gt == y_out and m_gt == m_out:
        return "year_month_match", 0.8

    # Year + month match after swap
    if int(d_out) <= 12 and y_gt == y_out and m_gt == d_out:
        return "year_month_swap_match", 0.8

    # Year only match
    if y_gt == y_out:
        return "year_match", 0.7

    return "no_match", 0.0


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows.")

    results = []
    for i, row in enumerate(rows):
        match_type, score = evaluate_dob(row["gt"], row["llm_output"])

        results.append({
            "Q_number":   row["Q_number"],
            "LoE":        row["LoE"],
            "LoQ":        row["LoQ"],
            "property":   row["property"],
            "gt":         row["gt"],
            "llm_output": row["llm_output"],
            "match_type": match_type,
            "score":      score,
        })

    # Write output
    fieldnames = ["Q_number", "LoE", "LoQ", "property", "gt", "llm_output", "match_type", "score"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    from collections import Counter
    print("\n=== Summary ===")
    match_counts = Counter(r["match_type"] for r in results)
    for mt, cnt in match_counts.most_common():
        scores = {
            "exact_match":            1.0,
            "swap_match":             1.0,
            "year_month_match":       0.8,
            "year_month_swap_match":  0.8,
            "year_match":             0.7,
            "no_match":               0.0,
        }
        print(f"  {mt:28s} {cnt:5d}  (score={scores[mt]})")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
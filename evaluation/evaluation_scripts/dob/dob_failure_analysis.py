"""
DoB Failure Mode Analysis for WILA-PopQA
Analyzes why Date of Birth scores are near-zero across all models.
Usage: python dob_failure_analysis.py --input_dir <path_to_dob_csvs>
"""
import pandas as pd
import glob
import os
import re
import argparse

def safe_year(s):
    m = re.match(r'^(\d{4})', str(s))
    return int(m.group(1)) if m else None

def classify_failure(output):
    output = str(output).strip()
    if output in ('YYYY-MM-DD', 'RRRR-MM-DD'):
        return 'literal_format_echo'
    if re.match(r'^\d{4}$', output):
        return 'year_only_no_format'
    if re.match(r'^\d{4}-\d{2}-\d{2}$', output):
        return 'wrong_date_correct_format'
    if len(output) > 40:
        return 'verbose_response'
    if output in ('nan', '', 'None'):
        return 'empty/nan'
    return 'other_format'

def main(input_dir):
    files = glob.glob(os.path.join(input_dir, "property_dob_*.csv"))
    if not files:
        print(f"No property_dob_*.csv files found in {input_dir}")
        return

    print(f"Found {len(files)} model files\n")

    dfs = []
    for f in sorted(files):
        model = os.path.basename(f).replace("property_dob_", "").replace(".csv", "")
        df = pd.read_csv(f)
        df['model'] = model
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df['llm_output'] = all_df['llm_output'].astype(str)

    # --- 1. Overall score distribution ---
    n = len(all_df)
    print("=" * 60)
    print("1. OVERALL SCORE DISTRIBUTION")
    print("=" * 60)
    print(f"Total evaluations:             {n:>7}")
    print(f"Exact match (score=1.0):       {(all_df['eval_score']==1.0).sum():>7}  ({(all_df['eval_score']==1.0).mean()*100:.1f}%)")
    print(f"Year+month match (score=0.7):  {(all_df['eval_score']==0.7).sum():>7}  ({(all_df['eval_score']==0.7).mean()*100:.1f}%)")
    print(f"Year-only match (score=0.5):   {(all_df['eval_score']==0.5).sum():>7}  ({(all_df['eval_score']==0.5).mean()*100:.1f}%)")
    print(f"No match (score=0.0):          {(all_df['eval_score']==0.0).sum():>7}  ({(all_df['eval_score']==0.0).mean()*100:.1f}%)")

    # --- 2. Format compliance ---
    print(f"\n{'=' * 60}")
    print("2. FORMAT COMPLIANCE (YYYY-MM-DD)")
    print("=" * 60)
    all_df['correct_format'] = all_df['llm_output'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)
    n_format = all_df['correct_format'].sum()
    n_format_wrong = ((all_df['correct_format']) & (all_df['eval_score'] == 0.0)).sum()
    n_format_correct = n_format - n_format_wrong
    print(f"Outputs in YYYY-MM-DD format:  {n_format:>7}  ({n_format / n * 100:.1f}%)")
    print(f"  of which correct date:       {n_format_correct:>7}  ({n_format_correct / n_format * 100:.1f}% of formatted)")
    print(f"  of which WRONG date:         {n_format_wrong:>7}  ({n_format_wrong / n_format * 100:.1f}% of formatted)")

    # --- 3. Failure mode classification ---
    print(f"\n{'=' * 60}")
    print("3. FAILURE MODE CLASSIFICATION (no_match cases only)")
    print("=" * 60)
    no_match = all_df[all_df['eval_type'] == 'no_match'].copy()
    no_match['failure_mode'] = no_match['llm_output'].apply(classify_failure)
    fc = no_match['failure_mode'].value_counts()
    for mode, count in fc.items():
        print(f"  {mode:<30} {count:>7}  ({count / len(no_match) * 100:.1f}%)")

    # --- 4. Year error analysis ---
    print(f"\n{'=' * 60}")
    print("4. YEAR ERROR ANALYSIS (wrong date, correct format)")
    print("=" * 60)
    wrong = no_match[no_match['llm_output'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)].copy()
    wrong['gt_year'] = wrong['gt'].apply(safe_year)
    wrong['llm_year'] = wrong['llm_output'].apply(safe_year)
    wrong = wrong.dropna(subset=['gt_year', 'llm_year'])
    wrong['year_diff'] = (wrong['llm_year'] - wrong['gt_year']).abs()

    print(f"Total cases: {len(wrong)}")
    print(f"Mean year diff:   {wrong['year_diff'].mean():.1f}")
    print(f"Median year diff: {wrong['year_diff'].median():.1f}")
    bins = [0, 1, 5, 10, 50, 100, 500, 5000]
    labels = ['0-1', '2-5', '6-10', '11-50', '51-100', '101-500', '500+']
    wrong['year_bucket'] = pd.cut(wrong['year_diff'], bins=bins, labels=labels, right=True)
    bc = wrong['year_bucket'].value_counts().sort_index()
    print(f"\nYear difference buckets:")
    for bucket, count in bc.items():
        print(f"  {bucket:<10} {count:>7}  ({count / len(wrong) * 100:.1f}%)")

    # --- 5. Per-model summary ---
    print(f"\n{'=' * 60}")
    print("5. PER-MODEL: FORMAT COMPLIANCE vs ACCURACY")
    print("=" * 60)
    print(f"  {'Model':<35} {'Format%':>8} {'MeanScore':>10}")
    print(f"  {'-'*55}")
    for model in sorted(all_df['model'].unique()):
        m = all_df[all_df['model'] == model]
        fmt = m['correct_format'].mean() * 100
        score = m['eval_score'].mean()
        print(f"  {model:<35} {fmt:>7.1f}% {score:>10.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoB Failure Mode Analysis")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing property_dob_*.csv files")
    args = parser.parse_args()
    main(args.input_dir)

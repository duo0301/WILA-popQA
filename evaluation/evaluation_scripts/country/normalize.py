"""
normalize.py — Translate non-English llm_output values to English.

Adds an `llm_output_en` column to country evaluation CSVs.

Usage:
    # Single file
    python normalize.py --input scored.csv --output normalized.csv --cache normalization_cache.json

    # Batch: process all property_country_*.csv in a directory
    python normalize.py --batch-dir path/to/model_dir --cache normalization_cache.json
    python normalize.py --batch-dir path/to/model_dir --output-dir path/to/out --cache normalization_cache.json
"""

import argparse
import csv
import glob
import json
import re
import os
import sys
from pathlib import Path

from openai import OpenAI

# ─── Deepseek client ─────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


TRANSLATE_SYSTEM_PROMPT = """\
You are a translation assistant for country and place names.

Given a non-English country/place name (or list of names), translate it to English.

Rules:
- If the input is a demonym or nationality adjective (e.g. "français", "русский", "deutsche"), \
return the corresponding **country name** in English (e.g. "France", "Russia", "Germany").
- If the input is a list (e.g. "[فرنسا, بولندا]"), return a JSON list of English names.
- If the input is "NONE" or empty, return "NONE".
- Respond with JSON only: {"en": "translated name"} or {"en": ["name1", "name2"]}
- Do not include any explanation, only the JSON object.
"""


def translate_to_english(client: OpenAI, text: str, loq: str) -> str:
    """Call Deepseek to translate a country/place name to English."""
    user_msg = f"Language code: {loq}\nTranslate to English: {text}"

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        en_value = result.get("en", text)
        # If it's a list, format as string representation
        if isinstance(en_value, list):
            return json.dumps(en_value, ensure_ascii=False)
        return str(en_value)
    except Exception as e:
        print(f"    Translation error (LoQ={loq}): {e}", file=sys.stderr)
        return text  # fallback: return original


# ─── Cache ────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def cache_key(loq: str, llm_output: str) -> str:
    return f"{loq}||{llm_output}"


def build_reverse_index(cache: dict) -> dict:
    """Build a reverse index: llm_output -> translated value, for language-agnostic lookups."""
    rev = {}
    for k, v in cache.items():
        if "||" in k:
            output_part = k.split("||", 1)[1]
            if output_part not in rev:
                rev[output_part] = v
    return rev


def try_resolve_list(llm_output: str, rev_index: dict) -> str | None:
    """If llm_output looks like a list (e.g. '[Norvège, Allemagne]'),
    try to resolve each element individually from the cache."""
    stripped = llm_output.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return None
    # Split by comma, but respect items that might have commas inside quotes
    items = [item.strip().strip("'\"") for item in inner.split(",")]
    items = [item for item in items if item]
    if not items:
        return None

    translated_items = []
    for item in items:
        # Try the item directly
        val = rev_index.get(item)
        if val is None:
            # Try with bracket wrapper (some cache entries are stored as "[X]")
            val = rev_index.get(f"[{item}]")
        if val is not None:
            # Unwrap if the cached value is itself a JSON list like '["France"]'
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list) and len(parsed) == 1:
                    translated_items.append(parsed[0])
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            translated_items.append(val)
        else:
            return None  # can't resolve all items, give up

    return json.dumps(translated_items, ensure_ascii=False)


def cache_lookup(cache: dict, rev_index: dict, loq: str, llm_output: str) -> str | None:
    """Look up a translation in cache — first try exact key,
    then language-agnostic reverse index, then parse as list."""
    exact = cache.get(cache_key(loq, llm_output))
    if exact is not None:
        return exact
    found = rev_index.get(llm_output)
    if found is not None:
        return found
    # Try resolving list elements individually
    return try_resolve_list(llm_output, rev_index)


# ─── Processing ──────────────────────────────────────────────────────────

def normalize_file(input_path: str, output_path: str, cache: dict, cache_path: str):
    """Process a single CSV: add llm_output_en column."""
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "llm_output_en" in fieldnames:
        print(f"  Skipping {input_path} -- already has llm_output_en column.")
        return

    print(f"Processing {input_path} ({len(rows)} rows)...", flush=True)

    # Build reverse index for language-agnostic cache lookups
    rev_index = build_reverse_index(cache)

    # Pass 1: resolve from cache, collect unique missing llm_output values
    missing_rows = []  # (row_index, loq, llm_output) for rows needing translation
    unique_missing = {}  # llm_output -> first loq seen (for the API call)

    for i, row in enumerate(rows):
        loq = row.get("LoQ", "")
        llm_output = row.get("llm_output", "")

        if loq == "en":
            row["llm_output_en"] = llm_output
        elif not llm_output or llm_output.strip() == "" or llm_output.strip().upper() == "NONE":
            row["llm_output_en"] = "NONE"
        else:
            cached = cache_lookup(cache, rev_index, loq, llm_output)
            if cached is not None:
                row["llm_output_en"] = cached
                cache[cache_key(loq, llm_output)] = cached
            else:
                missing_rows.append(i)
                if llm_output not in unique_missing:
                    unique_missing[llm_output] = loq

    unique_to_translate = list(unique_missing.items())  # [(llm_output, loq), ...]
    print(f"  {len(unique_to_translate)} unique translations needed ({len(missing_rows)} rows), "
          f"{len(rows) - len(missing_rows)} resolved from cache/English.", flush=True)

    # Prepare output fieldnames and write helper
    out_fieldnames = fieldnames + ["llm_output_en"]

    def write_output():
        """Write current state of all rows to output CSV."""
        # Backfill missing rows from rev_index before writing
        for j in missing_rows:
            llm_out = rows[j].get("llm_output", "")
            val = rev_index.get(llm_out)
            if val is not None:
                rows[j]["llm_output_en"] = val
            elif rows[j].get("llm_output_en") is None:
                rows[j]["llm_output_en"] = ""  # placeholder for not-yet-translated
        with open(output_path, "w", newline="", encoding="utf-8") as fw:
            writer = csv.DictWriter(fw, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Pass 2: translate unique values
    if unique_to_translate:
        client = get_client()

        for idx, (llm_output, loq) in enumerate(unique_to_translate):
            translated = translate_to_english(client, llm_output, loq)
            cache[cache_key(loq, llm_output)] = translated
            rev_index[llm_output] = translated

            try:
                print(f"    [{idx + 1}/{len(unique_to_translate)}] {loq}  {llm_output!r}  ->  {translated}", flush=True)
            except UnicodeEncodeError:
                print(f"    [{idx + 1}/{len(unique_to_translate)}] {loq}  (non-printable)  ->  {translated}", flush=True)

            if (idx + 1) % 100 == 0:
                save_cache(cache, cache_path)
                write_output()
                print(f"    ** Cache + CSV saved ({len(cache)} cache entries) **", flush=True)

        save_cache(cache, cache_path)
        print(f"  Translations done. Cache now has {len(cache)} entries.", flush=True)

    # Final write
    write_output()
    print(f"  Written to {output_path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normalize country evaluation CSVs by translating llm_output to English."
    )

    # Single-file mode
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file path (single-file mode)")

    # Batch mode
    parser.add_argument("--batch-dir", help="Directory containing property_country_*.csv files")
    parser.add_argument("--output-dir", help="Output directory for batch mode (default: same as batch-dir)")

    # Cache
    parser.add_argument("--cache", default="normalization_cache.json", help="Path to translation cache JSON")

    args = parser.parse_args()

    if not args.input and not args.batch_dir:
        parser.error("Either --input or --batch-dir is required.")

    if args.input and args.batch_dir:
        parser.error("Use either --input or --batch-dir, not both.")

    # Load cache
    cache = load_cache(args.cache)
    print(f"Loaded cache with {len(cache)} entries from {args.cache}")

    if args.input:
        # Single-file mode
        if not args.output:
            parser.error("--output is required in single-file mode.")
        normalize_file(args.input, args.output, cache, args.cache)

    elif args.batch_dir:
        # Batch mode
        pattern = os.path.join(args.batch_dir, "property_country_*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"No files matching {pattern}")
            sys.exit(1)

        output_dir = args.output_dir or args.batch_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Found {len(files)} files to process.")
        for filepath in files:
            basename = os.path.basename(filepath)
            out_path = os.path.join(output_dir, basename)
            normalize_file(filepath, out_path, cache, args.cache)

    print("\nAll done.")


if __name__ == "__main__":
    main()

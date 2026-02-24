"""
evaluate_country.py  (Step 2 — Deepseek V3 + Normalized English)

Usage:
    python evaluate_country.py --input normalized.csv --output eval.csv --cache vocab_cache.json
    python evaluate_country.py --batch-dir model/normalized/ --output-dir model/eval/ --cache vocab_cache.json

--input      : path to a single normalized CSV (must have: Q_number, gt, llm_output_en, prompt)
--output     : path for the evaluated output CSV
--batch-dir  : directory of normalized CSVs to process in batch
--output-dir : directory for batch output CSVs
--cache      : path to the vocabulary cache JSON (created if missing)
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote as url_encode

import requests
from openai import OpenAI

# ─── Deepseek client ────────────────────────────────────────────────────

ds_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ─── Configuration ───────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
TARGET_LANGUAGES = ["en"]

SPARQL_HEADERS = {"User-Agent": "ravenclaw-eval/0.1 (contact: andschimmenti@gmail.com)"}

SCORE_MAP = {
    "exact_match":      1.0,
    "alias_match":      0.9,
    "substring_match":  0.85,
    "demonym_match":    0.8,
    "historical_match": 0.75,
    "no_match":         0.0,
}


def safe_print(*args, **kwargs):
    """Print with flush=True and handle UnicodeEncodeError."""
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


# ─── Wikidata queries ───────────────────────────────────────────────────

def sparql_query(query: str, max_retries: int = 5) -> list[dict]:
    for attempt in range(max_retries):
        time.sleep(0.5)
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=SPARQL_HEADERS,
                timeout=30,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 10 * (attempt + 1)
            safe_print(f"  Connection error (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5)) * (attempt + 1)
            safe_print(f"  Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
    raise RuntimeError(f"SPARQL query failed after {max_retries} retries")


def fetch_country_of_citizenship(entity_q: str) -> list[str]:
    query = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?country WHERE {{
      wd:{entity_q} wdt:P27 ?country .
    }}
    """
    results = sparql_query(query)
    return [r["country"]["value"].split("/")[-1] for r in results]


def fetch_demonyms_and_aliases(country_q: str) -> dict[str, dict]:
    lang_filter = ", ".join(f'"{l}"' for l in TARGET_LANGUAGES)
    query = f"""
    PREFIX wd:   <http://www.wikidata.org/entity/>
    PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?val ?lang ?type WHERE {{
      BIND(wd:{country_q} AS ?country)
      {{
        ?country wdt:P1549 ?val .
        BIND("demonym" AS ?type)
      }} UNION {{
        ?country skos:altLabel ?val .
        BIND("alias" AS ?type)
      }} UNION {{
        ?country rdfs:label ?val .
        BIND("label" AS ?type)
      }}
      BIND(LANG(?val) AS ?lang)
      FILTER(?lang IN ({lang_filter}))
    }}
    """
    results = sparql_query(query)

    vocab: dict[str, dict] = {lang: {"demonyms": [], "aliases": [], "labels": []} for lang in TARGET_LANGUAGES}
    for r in results:
        lang = r["lang"]["value"]
        val  = r["val"]["value"]
        typ  = r["type"]["value"]
        if lang in vocab:
            key = {"demonym": "demonyms", "alias": "aliases", "label": "labels"}[typ]
            if val not in vocab[lang][key]:
                vocab[lang][key].append(val)

    return vocab


def fetch_successors(country_q: str) -> list[str]:
    query = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?terminal WHERE {{
      wd:{country_q} wdt:P1366+ ?terminal .
      FILTER NOT EXISTS {{ ?terminal wdt:P1366 ?next . }}
    }}
    """
    results = sparql_query(query)
    return [r["terminal"]["value"].split("/")[-1] for r in results]


def _has_any_demonym(vocab: dict) -> bool:
    return any(
        len(lang_data.get("demonyms", [])) > 0
        for lang_data in vocab.values()
    )


def _merge_vocab(target: dict, source: dict):
    for lang in TARGET_LANGUAGES:
        for key in ("demonyms", "aliases", "labels"):
            for item in source.get(lang, {}).get(key, []):
                if item not in target.get(lang, {}).get(key, []):
                    target.setdefault(lang, {}).setdefault(key, []).append(item)


def get_vocab_for_country(country_q: str, cache: dict, cache_path: str) -> dict:
    if country_q in cache:
        return cache[country_q]

    safe_print(f"  Fetching vocab for {country_q}...")
    vocab = fetch_demonyms_and_aliases(country_q)

    if not _has_any_demonym(vocab):
        successors = fetch_successors(country_q)
        if successors:
            safe_print(f"    No demonyms on {country_q}, walking to successors: {successors}")
            for succ_q in successors:
                succ_vocab = get_vocab_for_country(succ_q, cache, cache_path)
                _merge_vocab(vocab, succ_vocab)

    cache[country_q] = vocab
    save_cache(cache, cache_path)
    return vocab


# ─── Cache ───────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ─── Normalization ───────────────────────────────────────────────────────

def normalize_output(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []

    s = raw.strip()

    # Strip outer list syntax
    s = re.sub(r"^\[?\s*['\"]?", "", s)
    s = re.sub(r"['\"]?\s*\]?$", "", s)
    s = s.strip()

    # Remove trailing parenthetical
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()

    # Entity name leakage: "Entity Name - answer"
    if " - " in s:
        parts = s.split(" - ", 1)
        before, after = parts[0].strip(), parts[1].strip()
        if len(before) > len(after):
            s = after
        s = re.sub(r"\s*\[.*?\]\s*$", "", s).strip()

    # Split on comma
    raw_candidates = [c.strip() for c in s.split(",")]

    candidates = []
    for c in raw_candidates:
        c = re.sub(r"['\"\[\]\(\)]+", "", c).strip()
        c = re.sub(r"\s*\([^)]*\)\s*$", "", c).strip()
        if "/" in c:
            c = c.split("/")[0].strip()
        if c:
            candidates.append(c)

    return candidates


def parse_gt(gt_raw: str) -> list[str]:
    try:
        return ast.literal_eval(gt_raw)
    except (ValueError, SyntaxError):
        return [gt_raw.strip().strip("[]'\"")] if gt_raw.strip() else []


# ─── Matching ────────────────────────────────────────────────────────────

def casefold(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.casefold()).strip()


def match(candidates: list[str], gt: list[str], vocab: dict) -> tuple[str, float]:
    """
    Match candidates against GT + expanded English vocabulary.
    Returns (match_type, score).
    """
    gt_cf = [casefold(g) for g in gt]

    lang_vocab = vocab.get("en", {"demonyms": [], "aliases": [], "labels": []})
    aliases_cf  = [casefold(a) for a in lang_vocab.get("aliases", []) + lang_vocab.get("labels", [])]
    demonyms_cf = [casefold(d) for d in lang_vocab.get("demonyms", [])]

    best_type  = "no_match"
    best_score = 0.0

    for c in candidates:
        c_cf = casefold(c)
        if not c_cf:
            continue

        if c_cf in gt_cf:
            return "exact_match", SCORE_MAP["exact_match"]

        if c_cf in aliases_cf:
            if best_score < SCORE_MAP["alias_match"]:
                best_type, best_score = "alias_match", SCORE_MAP["alias_match"]

        if len(c_cf) >= 4:
            for g_cf in gt_cf:
                if len(g_cf) >= 4 and (c_cf in g_cf or g_cf in c_cf):
                    if best_score < SCORE_MAP["substring_match"]:
                        best_type, best_score = "substring_match", SCORE_MAP["substring_match"]

        if c_cf in demonyms_cf:
            if best_score < SCORE_MAP["demonym_match"]:
                best_type, best_score = "demonym_match", SCORE_MAP["demonym_match"]

    return best_type, best_score


# ─── LLM-as-a-Judge (Deepseek V3 fallback) ──────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. You must decide how an LLM's answer relates to the \
ground-truth answer for a "country of citizenship" question.

Classify into exactly one of these categories:

- "exact_match": The answer is essentially the same as the ground truth.
- "alias_match": The answer is a recognised alternate name or translation of the \
  correct country. E.g. GT="United States of America", answer="USA".
- "substring_match": One string contains the other (partial overlap). \
  E.g. GT="Kingdom of Spain", answer="Spain".
- "demonym_match": The answer is the nationality/demonym form instead of the country \
  name. E.g. GT="France", answer="French".
- "historical_match": The answer refers to a historically related state: a predecessor, \
  successor, or the modern/historical country whose territory largely overlaps with the \
  correct answer. E.g. GT="Turkey", answer="Ottoman Empire". \
  GT="Russia", answer="Soviet Union".
- "no_match": The answer is wrong, unrelated, or gibberish.

IMPORTANT: Use the original question for context about the historical figure being asked \
about. This helps determine whether the answer is a plausible historical/geographic match.

Respond with JSON: {"match_type": "<one of the six categories>", "reason": "<brief explanation>"}
"""


def _judge_cache_key(llm_output: str, gt_raw: str) -> str:
    """Cache key for judge results: normalized output + raw GT."""
    return f"{llm_output.strip()}|||{gt_raw.strip()}"


def llm_judge(llm_output: str, gt: list[str], gt_raw: str, vocab: dict, prompt: str,
              judge_cache: dict, judge_cache_path: str) -> tuple[str, float]:
    """
    Ask Deepseek V3 to classify a no_match row.
    Returns (match_type, score). Uses judge_cache to avoid repeat calls.
    """
    cache_key = _judge_cache_key(llm_output, gt_raw)
    if cache_key in judge_cache:
        mt = judge_cache[cache_key]
        return mt, SCORE_MAP.get(mt, 0.0)

    lang_vocab = vocab.get("en", {})
    aliases = lang_vocab.get("aliases", []) + lang_vocab.get("labels", [])
    demonyms = lang_vocab.get("demonyms", [])

    user_msg = (
        f"Original question asked to the LLM:\n{prompt}\n\n"
        f"Ground truth (accepted answers): {gt}\n"
        f"Known aliases/labels: {aliases[:20]}\n"
        f"Known demonyms: {demonyms[:20]}\n"
        f"LLM answer: {llm_output}\n\n"
        f"Classify this answer."
    )

    try:
        resp = ds_client.chat.completions.create(
            model="deepseek-chat",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        result = json.loads(resp.choices[0].message.content)
        mt = result.get("match_type", "no_match")
        if mt not in SCORE_MAP:
            mt = "no_match"
        judge_cache[cache_key] = mt
        save_cache(judge_cache, judge_cache_path)
        return mt, SCORE_MAP[mt]
    except Exception as e:
        safe_print(f"    Judge error: {e}")
        return "no_match", 0.0


# ─── Evaluation core ────────────────────────────────────────────────────

def evaluate_file(input_path: str, output_path: str, cache: dict, cache_path: str,
                  judge_cache: dict, judge_cache_path: str):
    """Evaluate a single normalized CSV and write results."""
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        rows = list(reader)
    safe_print(f"Loaded {len(rows)} rows from {input_path}")

    # Check for resume: if output already exists, load completed rows
    completed = {}
    if Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["Q_number"], r.get("LoQ", ""), r.get("LoE", ""))
                completed[key] = r
        safe_print(f"  Resuming: {len(completed)} rows already evaluated")

    # Output fieldnames = input fields + eval + eval_type
    out_fieldnames = list(input_fieldnames) + ["eval", "eval_type", "provenance"]

    results = []
    for i, row in enumerate(rows):
        entity_q = row["Q_number"]
        gt_raw   = row["gt"]
        output_en = row.get("llm_output_en", "")

        # Check if already evaluated
        key = (entity_q, row.get("LoQ", ""), row.get("LoE", ""))
        if key in completed:
            results.append(completed[key])
            continue

        safe_print(f"[{i+1}/{len(rows)}] {entity_q}")

        # Step 1: Check for empty output → FN
        is_empty = (
            not output_en
            or not output_en.strip()
            or output_en.strip().upper() in ("NONE", "N/A", "NA", "NULL", "[]", "['']", '[""]')
        )
        if is_empty:
            out_row = dict(row)
            out_row["eval"] = "FN"
            out_row["eval_type"] = ""
            out_row["provenance"] = ""
            results.append(out_row)
            if (i + 1) % 100 == 0:
                _save_results(output_path, out_fieldnames, results)
                save_cache(cache, cache_path)
                save_cache(judge_cache, judge_cache_path)
            continue

        # Step 2: Extract candidates from llm_output_en
        candidates = normalize_output(output_en)
        if not candidates:
            out_row = dict(row)
            out_row["eval"] = "FN"
            out_row["eval_type"] = ""
            out_row["provenance"] = ""
            results.append(out_row)
            if (i + 1) % 100 == 0:
                _save_results(output_path, out_fieldnames, results)
                save_cache(cache, cache_path)
                save_cache(judge_cache, judge_cache_path)
            continue

        # Step 3: Parse GT, fetch English vocab
        gt = parse_gt(gt_raw)

        if entity_q in cache.get("_citizenship", {}):
            country_qs = cache["_citizenship"][entity_q]
        else:
            country_qs = fetch_country_of_citizenship(entity_q)
            cache.setdefault("_citizenship", {})[entity_q] = country_qs
            save_cache(cache, cache_path)

        merged_vocab: dict[str, dict] = {"en": {"demonyms": [], "aliases": [], "labels": []}}
        for cq in country_qs:
            v = get_vocab_for_country(cq, cache, cache_path)
            for key in ("demonyms", "aliases", "labels"):
                for item in v.get("en", {}).get(key, []):
                    if item not in merged_vocab["en"][key]:
                        merged_vocab["en"][key].append(item)

        # Step 4: Match
        match_type, score = match(candidates, gt, merged_vocab)

        # Step 5: LLM judge fallback
        used_judge = False
        if match_type == "no_match" and candidates:
            match_type, score = llm_judge(
                output_en, gt, gt_raw, merged_vocab, row.get("prompt", ""),
                judge_cache, judge_cache_path,
            )
            used_judge = True
            if match_type != "no_match":
                safe_print(f"    Judge overrode -> {match_type}")

        # Step 6-7: Assign eval/eval_type
        out_row = dict(row)
        if match_type != "no_match":
            out_row["eval"] = "TP"
            out_row["eval_type"] = match_type
        else:
            out_row["eval"] = "FP"
            out_row["eval_type"] = "no_match"
        out_row["provenance"] = "judge" if used_judge else "auto"

        results.append(out_row)

        # Periodic save
        if (i + 1) % 100 == 0:
            _save_results(output_path, out_fieldnames, results)
            save_cache(cache, cache_path)
            save_cache(judge_cache, judge_cache_path)
            safe_print(f"  [checkpoint at row {i+1}]")

    # Final save
    _save_results(output_path, out_fieldnames, results)
    save_cache(cache, cache_path)
    save_cache(judge_cache, judge_cache_path)

    # Summary
    safe_print(f"\n=== Summary for {Path(input_path).name} ===")
    eval_counts = Counter(r["eval"] for r in results)
    for ev in ("TP", "FP", "FN"):
        safe_print(f"  {ev}: {eval_counts.get(ev, 0)}")
    type_counts = Counter(r["eval_type"] for r in results if r["eval"] == "TP")
    if type_counts:
        safe_print("  TP breakdown:")
        for mt, cnt in type_counts.most_common():
            safe_print(f"    {mt:22s} {cnt:5d}")
    safe_print(f"Results written to {output_path}")


def _save_results(output_path: str, fieldnames: list[str], results: list[dict]):
    """Write results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate country (Step 2 — Deepseek V3 + English)")
    parser.add_argument("--input",      help="Single normalized CSV to evaluate")
    parser.add_argument("--output",     help="Output CSV path (for single file mode)")
    parser.add_argument("--batch-dir",  help="Directory of normalized CSVs to batch-process")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    parser.add_argument("--cache",      default="vocab_cache.json", help="Vocabulary cache JSON path")
    parser.add_argument("--judge-cache", default="judge_cache.json", help="Judge results cache JSON path")
    args = parser.parse_args()

    # Validate args
    if args.batch_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --batch-dir")
    elif args.input:
        if not args.output:
            parser.error("--output is required with --input")
    else:
        parser.error("Either --input or --batch-dir is required")

    # Load caches
    cache = load_cache(args.cache)
    safe_print(f"Vocab cache has {len(cache)} entries.")
    judge_cache = load_cache(args.judge_cache)
    safe_print(f"Judge cache has {len(judge_cache)} entries.")

    if args.batch_dir:
        # Batch mode
        batch_dir = Path(args.batch_dir)
        output_dir = Path(args.output_dir)
        csvs = sorted(batch_dir.glob("property_country_*.csv"))
        if not csvs:
            csvs = sorted(batch_dir.glob("*.csv"))
        safe_print(f"Found {len(csvs)} CSV files in {batch_dir}")
        for csv_path in csvs:
            out_path = output_dir / (csv_path.stem + "_prov" + csv_path.suffix)
            safe_print(f"\n{'='*60}")
            safe_print(f"Processing: {csv_path.name}")
            safe_print(f"{'='*60}")
            evaluate_file(str(csv_path), str(out_path), cache, args.cache,
                         judge_cache, args.judge_cache)
    else:
        # Single file mode
        out_path = Path(args.output)
        out_path = out_path.with_name(out_path.stem + "_prov" + out_path.suffix)
        evaluate_file(args.input, str(out_path), cache, args.cache,
                     judge_cache, args.judge_cache)


if __name__ == "__main__":
    main()

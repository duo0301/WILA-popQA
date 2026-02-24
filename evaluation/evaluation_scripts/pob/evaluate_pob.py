"""
evaluate_pob.py — Fast POB evaluation (Deepseek V3 judge, parallel)

Two-pass approach:
  1. String matching + Wikidata resolution (cached, 1 query per unique place)
  2. Parallel LLM judge for remaining no_match rows

Usage:
    python evaluate_pob.py --input normalized.csv --output eval.csv --cache pob_cache.json
    python evaluate_pob.py --batch-dir model/normalized/ --output-dir model/eval/ --cache pob_cache.json
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from openai import OpenAI

# ─── Deepseek client ────────────────────────────────────────────────────

ds_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ─── Configuration ───────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WB_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "ravenclaw-eval/0.1 (contact: andschimmenti@gmail.com)"}

SCORES = {
    "exact_match":      1.0,
    "country_match":    0.8,
    "historical_match": 0.7,
    "llm_match":        0.65,
    "no_match":         0.0,
}


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


# ─── Cache ───────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ─── Wikidata (simplified: 1 search + 1 SPARQL per place) ───────────────

def search_entity(name: str, lang: str = "en") -> str | None:
    """Resolve a place name to a single Q-number via wbsearchentities."""
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": lang,
        "type": "item",
        "limit": 1,
        "format": "json",
    }
    for attempt in range(3):
        try:
            resp = requests.get(WB_API_ENDPOINT, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("search", [])
            return results[0]["id"] if results else None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(5 * (attempt + 1))
        except Exception:
            return None
    return None


def fetch_country(q: str) -> list[str]:
    """Get P17 (country) for a Q-number. Returns list of country Q-numbers."""
    query = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?country WHERE {{ wd:{q} wdt:P17 ?country . }}
    """
    for attempt in range(3):
        try:
            time.sleep(0.1)
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 5)) * (attempt + 1))
                continue
            resp.raise_for_status()
            return [r["country"]["value"].split("/")[-1] for r in resp.json()["results"]["bindings"]]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(5 * (attempt + 1))
        except Exception:
            return []
    return []


def resolve_place(name: str, cache: dict) -> dict:
    """
    Resolve a place name (English) to {q: str|None, countries: [str]}.
    Cached — same name never queried twice.
    """
    key = f"place_en:{name.strip().lower()}"
    if key in cache:
        return cache[key]

    safe_print(f"  Resolving '{name}'...")
    q = search_entity(name, "en")
    countries = []
    if q:
        countries = fetch_country(q)

    result = {"q": q, "countries": countries}
    cache[key] = result
    return result


# ─── Normalization ───────────────────────────────────────────────────────

def clean_output(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = s.split("\n")[0].strip()
    s = re.sub(r"['\"\[\]]+", "", s).strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = s.split(",")[0].strip()
    s = s.rstrip(".").strip()
    return s


def parse_gt(gt_raw: str) -> list[str]:
    try:
        val = ast.literal_eval(gt_raw)
        if isinstance(val, list):
            return val
        return [str(val)]
    except (ValueError, SyntaxError):
        return [gt_raw.strip().strip("[]'\"")] if gt_raw.strip() else []


def casefold(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.casefold()).strip()


# ─── LLM Judge ───────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. You must decide how an LLM's answer relates to the \
ground-truth answer for a "place of birth" question.

Classify into exactly one of these categories:

- "exact_match": The answer is essentially the same place as the ground truth, \
  possibly in a different language or transliteration. \
  E.g. GT="Moscow", answer="Москва". GT="Munich", answer="München".
- "country_match": The answer is in the same country as the ground truth but is a \
  different place (e.g. different city). Or the answer is the country itself when \
  the GT is a city within that country. \
  E.g. GT="Lyon", answer="France". GT="Kyoto", answer="Tokyo" (both in Japan).
- "historical_match": The answer refers to a historically related place — the same \
  location but under a different historical name or political entity. \
  E.g. GT="Istanbul", answer="Constantinople". GT="St. Petersburg", answer="Leningrad".
- "no_match": The answer is wrong, unrelated, or gibberish.

IMPORTANT: Use the original question for context about the person being asked about.

Respond with JSON: {"match_type": "<one of the four categories>", "reason": "<brief explanation>"}
"""


def _judge_cache_key(llm_output: str, gt_raw: str) -> str:
    return f"{llm_output.strip()}|||{gt_raw.strip()}"


def call_judge_single(llm_output: str, gt_raw: str, prompt: str) -> str:
    """Single Deepseek judge call. Returns match_type string."""
    user_msg = (
        f"Original question asked to the LLM:\n{prompt}\n\n"
        f"Ground truth (accepted answer): {gt_raw}\n"
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
        return mt if mt in SCORES else "no_match"
    except Exception as e:
        safe_print(f"    Judge error: {e}")
        return "no_match"


def batch_judge(no_match_items: list[dict], judge_cache: dict, judge_cache_path: str):
    """
    Parallel judge calls for no_match rows.
    Each item: {index, llm_output, gt_raw, prompt}
    Updates judge_cache in place and returns {index: match_type}.
    """
    # Filter out already-cached
    to_judge = []
    results = {}
    for item in no_match_items:
        ck = _judge_cache_key(item["llm_output"], item["gt_raw"])
        if ck in judge_cache:
            results[item["index"]] = judge_cache[ck]
        else:
            to_judge.append(item)

    cached_count = len(results)
    safe_print(f"  Judge: {len(to_judge)} to call, {cached_count} from cache")

    if not to_judge:
        return results

    lock = threading.Lock()
    completed = [0]

    def do_judge(item):
        mt = call_judge_single(item["llm_output"], item["gt_raw"], item["prompt"])
        return item, mt

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(do_judge, item): item for item in to_judge}

        for future in as_completed(futures):
            item, mt = future.result()
            ck = _judge_cache_key(item["llm_output"], item["gt_raw"])

            with lock:
                judge_cache[ck] = mt
                results[item["index"]] = mt
                completed[0] += 1
                count = completed[0]

            if mt != "no_match":
                safe_print(f"    [{count}/{len(to_judge)}] Judge -> {mt}")

            if count % 200 == 0:
                with lock:
                    save_cache(judge_cache, judge_cache_path)
                safe_print(f"    [judge checkpoint at {count}]")

    save_cache(judge_cache, judge_cache_path)
    safe_print(f"  Judge done: {len(to_judge)} calls completed")
    return results


# ─── Evaluation core ────────────────────────────────────────────────────

def evaluate_file(input_path: str, output_path: str,
                  cache: dict, cache_path: str,
                  judge_cache: dict, judge_cache_path: str):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        rows = list(reader)
    safe_print(f"Loaded {len(rows)} rows from {input_path}")

    # Resume support
    completed = {}
    if Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("eval"):  # only count rows that have been evaluated
                    key = (r["Q_number"], r.get("LoQ", ""), r.get("LoE", ""))
                    completed[key] = r
        if completed:
            safe_print(f"  Resuming: {len(completed)} rows already evaluated")

    out_fieldnames = list(input_fieldnames) + ["eval", "eval_type", "provenance"]

    # ── Pass 1: String matching + Wikidata resolution ──
    results = [None] * len(rows)
    no_match_for_judge = []

    for i, row in enumerate(rows):
        entity_q   = row["Q_number"]
        gt_raw     = row["gt"]
        output_en  = row.get("llm_output_en", "")

        # Check resume
        key = (entity_q, row.get("LoQ", ""), row.get("LoE", ""))
        if key in completed:
            results[i] = completed[key]
            continue

        # FN check
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
            results[i] = out_row
            continue

        out_clean = clean_output(output_en)
        if not out_clean:
            out_row = dict(row)
            out_row["eval"] = "FN"
            out_row["eval_type"] = ""
            out_row["provenance"] = ""
            results[i] = out_row
            continue

        # Parse GT
        gt_list = parse_gt(gt_raw)
        gt_cf = [casefold(g) for g in gt_list]
        out_cf = casefold(out_clean)

        # String exact match
        if out_cf and out_cf in gt_cf:
            out_row = dict(row)
            out_row["eval"] = "TP"
            out_row["eval_type"] = "exact_match"
            out_row["provenance"] = "auto"
            results[i] = out_row
            continue

        # String substring match (min 4 chars)
        substring_hit = False
        if len(out_cf) >= 4:
            for g_cf in gt_cf:
                if len(g_cf) >= 4 and (out_cf in g_cf or g_cf in out_cf):
                    substring_hit = True
                    break
        if substring_hit:
            out_row = dict(row)
            out_row["eval"] = "TP"
            out_row["eval_type"] = "exact_match"
            out_row["provenance"] = "auto"
            results[i] = out_row
            continue

        # Wikidata resolution: compare Q-numbers
        out_resolved = resolve_place(out_clean, cache)
        gt_resolved_list = [resolve_place(clean_output(g), cache) for g in gt_list]

        # Q-number match
        if out_resolved["q"]:
            for gt_res in gt_resolved_list:
                if gt_res["q"] and gt_res["q"] == out_resolved["q"]:
                    out_row = dict(row)
                    out_row["eval"] = "TP"
                    out_row["eval_type"] = "exact_match"
                    out_row["provenance"] = "auto"
                    results[i] = out_row
                    break
            if results[i] is not None:
                continue

        # Country match (P17)
        if out_resolved["countries"]:
            gt_countries = set()
            for gt_res in gt_resolved_list:
                gt_countries.update(gt_res.get("countries", []))
            if gt_countries and set(out_resolved["countries"]) & gt_countries:
                out_row = dict(row)
                out_row["eval"] = "TP"
                out_row["eval_type"] = "country_match"
                out_row["provenance"] = "auto"
                results[i] = out_row
                continue

        # Couldn't match — queue for judge
        no_match_for_judge.append({
            "index": i,
            "llm_output": output_en,
            "gt_raw": gt_raw,
            "prompt": row.get("prompt", ""),
        })
        # Placeholder
        results[i] = None

        # Periodic cache save
        if (i + 1) % 500 == 0:
            save_cache(cache, cache_path)
            safe_print(f"  [Wikidata pass: {i+1}/{len(rows)}]")

    save_cache(cache, cache_path)

    # Count pass 1 stats
    fn_count = sum(1 for r in results if r and r.get("eval") == "FN")
    tp_count = sum(1 for r in results if r and r.get("eval") == "TP")
    safe_print(f"\n  Pass 1 done: {tp_count} TP, {fn_count} FN, {len(no_match_for_judge)} -> judge")

    # ── Pass 2: Parallel LLM judge ──
    if no_match_for_judge:
        judge_results = batch_judge(no_match_for_judge, judge_cache, judge_cache_path)

        for item in no_match_for_judge:
            idx = item["index"]
            mt = judge_results.get(idx, "no_match")
            out_row = dict(rows[idx])
            if mt != "no_match":
                out_row["eval"] = "TP"
                out_row["eval_type"] = mt
            else:
                out_row["eval"] = "FP"
                out_row["eval_type"] = "no_match"
            out_row["provenance"] = "judge"
            results[idx] = out_row

    # Final save
    _save_results(output_path, out_fieldnames, results)

    # Summary
    safe_print(f"\n=== Summary for {Path(input_path).name} ===")
    eval_counts = Counter(r["eval"] for r in results if r)
    for ev in ("TP", "FP", "FN"):
        safe_print(f"  {ev}: {eval_counts.get(ev, 0)}")
    type_counts = Counter(r["eval_type"] for r in results if r and r.get("eval") == "TP")
    if type_counts:
        safe_print("  TP breakdown:")
        for mt, cnt in type_counts.most_common():
            safe_print(f"    {mt:22s} {cnt:5d}")
    safe_print(f"Results written to {output_path}")


def _save_results(output_path: str, fieldnames: list[str], results: list[dict]):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r:
                writer.writerow(r)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate POB (fast, parallel judge)")
    parser.add_argument("--input",       help="Single normalized CSV")
    parser.add_argument("--output",      help="Output CSV path")
    parser.add_argument("--batch-dir",   help="Directory of normalized CSVs")
    parser.add_argument("--output-dir",  help="Output directory for batch mode")
    parser.add_argument("--cache",       default="pob_place_cache.json", help="Place resolution cache")
    parser.add_argument("--judge-cache", default="pob_judge_cache.json", help="Judge results cache")
    args = parser.parse_args()

    if args.batch_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --batch-dir")
    elif args.input:
        if not args.output:
            parser.error("--output is required with --input")
    else:
        parser.error("Either --input or --batch-dir is required")

    cache = load_cache(args.cache)
    safe_print(f"Place cache has {len(cache)} entries.")
    judge_cache = load_cache(args.judge_cache)
    safe_print(f"Judge cache has {len(judge_cache)} entries.")

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        output_dir = Path(args.output_dir)
        csvs = sorted(batch_dir.glob("property_pob_*.csv"))
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
        out_path = Path(args.output)
        out_path = out_path.with_name(out_path.stem + "_prov" + out_path.suffix)
        evaluate_file(args.input, str(out_path), cache, args.cache,
                     judge_cache, args.judge_cache)


if __name__ == "__main__":
    main()

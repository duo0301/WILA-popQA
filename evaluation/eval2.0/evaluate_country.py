"""
evaluate_country.py

Usage:
    python evaluate_country.py --input scored.csv --output results.csv --cache vocab_cache.json

--input  : path to the scored CSV (must have columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
--output : path for the scored output CSV
--cache  : path to the vocabulary cache JSON (created if missing, updated on each new country fetch)
"""

import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote as url_encode

import requests

# ─── Configuration ────────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
TARGET_LANGUAGES = ["en", "fr", "de", "it", "ru", "pl", "ar", "zh", "hi"]

# P27 = country of citizenship
# P1549 = demonym
SPARQL_HEADERS = {"User-Agent": "ravenclaw-eval/0.1 (contact: andschimmenti@gmail.com)"}

SCORE_MAP = {
    "exact_match":     1.0,
    "alias_match":     0.9,
    "substring_match": 0.85,
    "demonym_match":   0.8,
    "no_match":        0.0,
}


# ─── Wikidata queries ─────────────────────────────────────────────────────

def sparql_query(query: str) -> list[dict]:
    """Send a SPARQL query, return list of result bindings. Retries once on 429."""
    while True:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=SPARQL_HEADERS,
            timeout=30,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]


def fetch_country_of_citizenship(entity_q: str) -> list[str]:
    """
    Given an entity Q-number, return the list of country Q-numbers
    from P27 (country of citizenship). Returns empty list if none found.
    """
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
    """
    Given a country Q-number, fetch P1549 (demonyms) and skos:altLabel (aliases)
    for all target languages. Returns:
        { lang: { "demonyms": [...], "aliases": [...] } }
    """
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


# ─── Cache ────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_successors(country_q: str) -> list[str]:
    """
    Walk the full P1366 (replaced by) chain from a historical state
    and return only terminal entities — those with no further P1366 out-edge.
    Uses transitive closure (wdt:P1366+) to resolve the entire chain in one query.
    e.g. Kingdom of Egypt → United Arab Republic → Republic of Egypt → Egypt (Q869)
    """
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
    """Check whether any language in the vocab has at least one demonym."""
    return any(
        len(lang_data.get("demonyms", [])) > 0
        for lang_data in vocab.values()
    )


def _merge_vocab(target: dict, source: dict):
    """Merge source vocab into target in place, deduplicating."""
    for lang in TARGET_LANGUAGES:
        for key in ("demonyms", "aliases", "labels"):
            for item in source.get(lang, {}).get(key, []):
                if item not in target.get(lang, {}).get(key, []):
                    target.setdefault(lang, {}).setdefault(key, []).append(item)


def get_vocab_for_country(country_q: str, cache: dict, cache_path: str) -> dict:
    """
    Return vocab for a country Q-number. Hits cache first;
    fetches from Wikidata and saves cache on miss.
    If the fetched vocab has no demonyms in any language,
    walks P1376 successors and merges their vocabs in.
    """
    if country_q in cache:
        return cache[country_q]

    print(f"  Fetching vocab for {country_q}...")
    vocab = fetch_demonyms_and_aliases(country_q)

    # If no demonyms found, this is likely a historical state.
    # Walk P1376 to successors and pull their vocabs.
    if not _has_any_demonym(vocab):
        successors = fetch_successors(country_q)
        if successors:
            print(f"    No demonyms on {country_q}, walking to successors: {successors}")
            for succ_q in successors:
                succ_vocab = get_vocab_for_country(succ_q, cache, cache_path)
                _merge_vocab(vocab, succ_vocab)

    cache[country_q] = vocab
    save_cache(cache, cache_path)
    return vocab


# ─── Normalization (Stage 0) ──────────────────────────────────────────────

def normalize_output(raw: str) -> list[str]:
    """
    Extract candidate answer strings from raw LLM output.
    Returns a list of cleaned candidates.
    """
    if not raw or not raw.strip():
        return []

    s = raw.strip()

    # Strip outer list syntax: ['...'], ["..."], [...]
    s = re.sub(r"^\[?\s*['\"]?", "", s)
    s = re.sub(r"['\"]?\s*\]?$", "", s)
    s = s.strip()

    # Remove trailing parenthetical: "answer (translation)"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()

    # Entity name leakage: "Entity Name - answer"
    # Heuristic: if the part before " - " is longer than the part after, it's the entity name
    if " - " in s:
        parts = s.split(" - ", 1)
        before, after = parts[0].strip(), parts[1].strip()
        if len(before) > len(after):
            s = after
        s = re.sub(r"\s*\[.*?\]\s*$", "", s).strip()

    # Split on comma → multiple candidates
    raw_candidates = [c.strip() for c in s.split(",")]

    candidates = []
    for c in raw_candidates:
        # Strip all bracket/quote chars (handles unmatched brackets too)
        c = re.sub(r"['\"\[\]\(\)]+", "", c).strip()
        # Strip parenthetical per candidate
        c = re.sub(r"\s*\([^)]*\)\s*$", "", c).strip()
        # Slash-separated alternates → take first
        if "/" in c:
            c = c.split("/")[0].strip()
        if c:
            candidates.append(c)

    return candidates


def parse_gt(gt_raw: str) -> list[str]:
    """Parse the GT field (a Python-literal list of strings)."""
    try:
        return ast.literal_eval(gt_raw)
    except (ValueError, SyntaxError):
        return [gt_raw.strip().strip("[]'\"")] if gt_raw.strip() else []


# ─── Matching (Stage 1) ───────────────────────────────────────────────────

def casefold(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.casefold()).strip()


def match(candidates: list[str], gt: list[str], vocab: dict, loq: str) -> tuple[str, float]:
    """
    Match candidates against GT + expanded vocabulary.
    Returns (match_type, score) for the best match found.

    Priority order: exact > alias > substring > demonym
    """
    gt_cf = [casefold(g) for g in gt]

    # Collect expanded sets from vocab for the prompt language
    lang_vocab = vocab.get(loq, {"demonyms": [], "aliases": [], "labels": []})
    aliases_cf  = [casefold(a) for a in lang_vocab.get("aliases", []) + lang_vocab.get("labels", [])]
    demonyms_cf = [casefold(d) for d in lang_vocab.get("demonyms", [])]

    best_type  = "no_match"
    best_score = 0.0

    for c in candidates:
        c_cf = casefold(c)
        if not c_cf:
            continue

        # 1. Exact match against GT
        if c_cf in gt_cf:
            return "exact_match", SCORE_MAP["exact_match"]

        # 2. Alias match
        if c_cf in aliases_cf:
            if best_score < SCORE_MAP["alias_match"]:
                best_type, best_score = "alias_match", SCORE_MAP["alias_match"]

        # 3. Substring match (either direction, min 4 chars)
        if len(c_cf) >= 4:
            for g_cf in gt_cf:
                if len(g_cf) >= 4 and (c_cf in g_cf or g_cf in c_cf):
                    if best_score < SCORE_MAP["substring_match"]:
                        best_type, best_score = "substring_match", SCORE_MAP["substring_match"]

        # 4. Demonym match
        if c_cf in demonyms_cf:
            if best_score < SCORE_MAP["demonym_match"]:
                best_type, best_score = "demonym_match", SCORE_MAP["demonym_match"]

    return best_type, best_score


# ─── Systematic flags ─────────────────────────────────────────────────────

def get_flag(loq: str) -> str | None:
    if loq == "zh":
        return "systematic_zh"
    if loq == "ar":
        return "systematic_ar"
    return None


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input scored CSV")
    parser.add_argument("--output", required=True, help="Output results CSV")
    parser.add_argument("--cache",  default="vocab_cache.json", help="Vocabulary cache JSON path")
    args = parser.parse_args()

    # Load input
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows.")

    # Load cache
    cache = load_cache(args.cache)
    print(f"Cache has {len(cache)} countries.")

    # Process
    results = []
    for i, row in enumerate(rows):
        entity_q = row["Q_number"]
        loq      = row["LoQ"]
        gt_raw   = row["gt"]
        output   = row["llm_output"]

        print(f"[{i+1}/{len(rows)}] {entity_q} LoQ={loq}")

        # Flag systematic failures
        flag = get_flag(loq)

        # Parse GT
        gt = parse_gt(gt_raw)

        # Fetch country of citizenship for this entity, then vocab for each country
        # Merge vocabs if multiple citizenships
        country_qs = fetch_country_of_citizenship(entity_q)
        merged_vocab: dict[str, dict] = {lang: {"demonyms": [], "aliases": [], "labels": []} for lang in TARGET_LANGUAGES}

        for cq in country_qs:
            v = get_vocab_for_country(cq, cache, args.cache)
            for lang in TARGET_LANGUAGES:
                for key in ("demonyms", "aliases", "labels"):
                    for item in v.get(lang, {}).get(key, []):
                        if item not in merged_vocab[lang][key]:
                            merged_vocab[lang][key].append(item)

        # Normalize output
        candidates = normalize_output(output)

        # Match
        match_type, score = match(candidates, gt, merged_vocab, loq)

        results.append({
            "Q_number":      entity_q,
            "LoE":           row["LoE"],
            "LoQ":           loq,
            "property":      row["property"],
            "gt":            gt_raw,
            "llm_output":    output,
            "candidates":    json.dumps(candidates, ensure_ascii=False),
            "country_Qs":    json.dumps(country_qs),
            "match_type":    match_type,
            "score":         score,
            "flag":          flag or "",
        })

    # Write output
    fieldnames = ["Q_number", "LoE", "LoQ", "property", "gt", "llm_output",
                  "candidates", "country_Qs", "match_type", "score", "flag"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    from collections import Counter
    print("\n=== Summary ===")
    match_counts = Counter(r["match_type"] for r in results)
    for mt, cnt in match_counts.most_common():
        print(f"  {mt:22s} {cnt:5d}  (score={SCORE_MAP.get(mt, 0.0)})")
    flag_counts = Counter(r["flag"] for r in results if r["flag"])
    if flag_counts:
        print("\n  Systematic flags:")
        for fl, cnt in flag_counts.most_common():
            print(f"    {fl:22s} {cnt:5d}")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
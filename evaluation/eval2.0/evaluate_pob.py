"""
evaluate_pob.py

Usage:
    python evaluate_pob.py --input scored.csv --output results_pob.csv --cache pob_cache.json

--input  : path to the CSV (columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
--output : path for the scored output CSV
--cache  : path to the cache JSON (created if missing, updated on each fetch)
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

# ─── Configuration ────────────────────────────────────────────────────────

SPARQL_ENDPOINT   = "https://query.wikidata.org/sparql"
WB_API_ENDPOINT   = "https://www.wikidata.org/w/api.php"
TARGET_LANGUAGES  = ["en", "fr", "de", "it", "ru", "pl", "ar", "zh", "hi"]
HEADERS           = {"User-Agent": "ravenclaw-eval/0.1 (contact: your-email@example.com)"}

SCORES = {
    "exact_match":     1.0,
    "country_match":   0.8,
    "historical_match": 0.7,
    "no_match":        0.0,
}


# ─── Cache ────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ─── Wikidata API calls ───────────────────────────────────────────────────

def sparql_query(query: str) -> list[dict]:
    """Send a SPARQL query, return list of result bindings. Retries on 429."""
    while True:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"    Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]


def search_entities(name: str, lang: str) -> list[str]:
    """
    Use wbsearchentities to resolve a place name string to Q-numbers.
    Returns top 5 results as a list of Q-numbers.
    """
    params = {
        "action":   "wbsearchentities",
        "search":   name,
        "language": lang,
        "type":     "item",
        "limit":    5,
        "format":   "json",
    }
    try:
        resp = requests.get(WB_API_ENDPOINT, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [r["id"] for r in data.get("search", [])]
    except Exception as e:
        print(f"    wbsearchentities error for '{name}': {e}")
    return []


def fetch_entity_info(q: str) -> dict:
    """
    For a given Q-number, fetch:
      - labels in all target languages
      - aliases in all target languages
      - P17 (country) values
      - P31 (instance of) values — to distinguish city vs country
    Single SPARQL query.
    """
    lang_filter = ", ".join(f'"{l}"' for l in TARGET_LANGUAGES)
    query = f"""
    PREFIX wd:   <http://www.wikidata.org/entity/>
    PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?label ?alias ?country ?instance
           (LANG(?label) AS ?label_lang) (LANG(?alias) AS ?alias_lang)
    WHERE {{
      BIND(wd:{q} AS ?entity)
      OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) IN ({lang_filter})) }}
      OPTIONAL {{ ?entity skos:altLabel ?alias . FILTER(LANG(?alias) IN ({lang_filter})) }}
      OPTIONAL {{ ?entity wdt:P17 ?country . }}
      OPTIONAL {{ ?entity wdt:P31 ?instance . }}
    }}
    """
    results = sparql_query(query)

    info = {
        "labels":    {},   # { lang: [str, ...] }
        "aliases":   {},
        "countries": set(),
        "instances": set(),
    }
    for r in results:
        if "label" in r and "label_lang" in r:
            lang = r["label_lang"]["value"]
            val  = r["label"]["value"]
            info["labels"].setdefault(lang, [])
            if val not in info["labels"][lang]:
                info["labels"][lang].append(val)
        if "alias" in r and "alias_lang" in r:
            lang = r["alias_lang"]["value"]
            val  = r["alias"]["value"]
            info["aliases"].setdefault(lang, [])
            if val not in info["aliases"][lang]:
                info["aliases"][lang].append(val)
        if "country" in r:
            info["countries"].add(r["country"]["value"].split("/")[-1])
        if "instance" in r:
            info["instances"].add(r["instance"]["value"].split("/")[-1])

    # Convert sets to lists for JSON serialisation
    info["countries"] = list(info["countries"])
    info["instances"] = list(info["instances"])
    return info


def fetch_country_set(country_q: str) -> list[str]:
    """
    Given a country Q-number, return the full set of countries a place
    has been part of across time. This requires walking in both directions:
      - Forward via P1366 (replaced by): what did this country become?
      - Backward via P1365 (replaces):   what did this country replace?
    Example: Israel (Q801)
      - P1366 forward: nothing (still exists)
      - P1365 backward: Mandatory Palestine -> Ottoman Syria -> ...
    Both chains are resolved transitively in a single query each.
    """
    query_forward_all = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?node WHERE {{ wd:{country_q} wdt:P1366+ ?node . }}
    """
    query_backward_all = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?node WHERE {{ wd:{country_q} wdt:P1365+ ?node . }}
    """

    result_set = {country_q}
    for query, key in [
        (query_forward_all,  "node"),
        (query_backward_all, "node"),
    ]:
        results = sparql_query(query)
        for r in results:
            result_set.add(r[key]["value"].split("/")[-1])

    return list(result_set)


# ─── Cached fetchers ──────────────────────────────────────────────────────

def get_entity_info(q: str, cache: dict, cache_path: str) -> dict:
    key = f"info:{q}"
    if key in cache:
        return cache[key]
    print(f"  Fetching info for {q}...")
    info = fetch_entity_info(q)
    cache[key] = info
    save_cache(cache, cache_path)
    return info


def get_country_set(country_q: str, cache: dict, cache_path: str) -> list[str]:
    key = f"country_set:{country_q}"
    if key in cache:
        return cache[key]
    print(f"  Fetching country set for {country_q}...")
    cset = fetch_country_set(country_q)
    cache[key] = cset
    save_cache(cache, cache_path)
    return cset


def resolve_entities(name: str, lang: str, cache: dict, cache_path: str) -> list[str]:
    """
    Resolve a place name to a list of candidate Q-numbers.
    Fetches top 5 from wbsearchentities, then filters to geographic entities only
    using P31. Results are cached.
    """
    key = f"resolve:{lang}:{name}"
    if key in cache:
        return cache[key]
    print(f"  Resolving '{name}' in {lang}...")
    candidates = search_entities(name, lang)
    geographic = []
    for q in candidates:
        info = get_entity_info(q, cache, cache_path)
        if set(info["instances"]) & GEOGRAPHIC_INSTANCES:
            geographic.append(q)
    cache[key] = geographic
    save_cache(cache, cache_path)
    return geographic


# ─── Normalization ────────────────────────────────────────────────────────

def clean_output(raw: str) -> str:
    """Strip common noise from LLM output for PoB."""
    if not raw:
        return ""
    s = raw.strip()
    # Take only first line (handles outputs with explanations)
    s = s.split("\n")[0].strip()
    # Strip brackets/quotes
    s = re.sub(r"['\"\[\]]+", "", s).strip()
    # Strip trailing parenthetical
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    # Strip trailing comma + anything after
    s = s.split(",")[0].strip()
    # Strip trailing period
    s = s.rstrip(".").strip()
    return s


# ─── Helpers ──────────────────────────────────────────────────────────────

# Known Q-numbers for "country" instance types
COUNTRY_INSTANCES = {
    "Q6256",    # country
    "Q3624078", # sovereign state
    "Q7275",    # state
    "Q788176",  # autonomous region
}

# Geographic/place instance types used to filter wbsearchentities results
GEOGRAPHIC_INSTANCES = COUNTRY_INSTANCES | {
    "Q515",         # city
    "Q2221906",     # geographic location
    "Q449111",      # subdivision
    "Q15284",       # municipality
    "Q532",         # village
    "Q3957",        # town
    "Q7930989",     # city or town
    "Q10354598",
    "Q124250988",   # urban settlement
    "Q486972",
    "Q123964505",   # populated place
    "Q1648",        # region
    "Q123705",      # district
    "Q34876",       # province
    "Q56061",       # administrative territorial entity
    "Q1620908",     # historical region
    "Q96088545",
    "Q52551684",
    "Q4835091",
    "Q82794",       # region
    "Q137023128",   # alt region
    "Q618123",      # geofeature
}


def is_country(info: dict) -> bool:
    """Check whether an entity's P31 values indicate it's a country/state."""
    return bool(set(info["instances"]) & COUNTRY_INSTANCES)


def collect_labels(info: dict, lang: str) -> list[str]:
    """Collect all labels and aliases for a given language."""
    out = []
    out.extend(info.get("labels", {}).get(lang, []))
    out.extend(info.get("aliases", {}).get(lang, []))
    return out


def casefold(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.casefold()).strip()


# ─── Matching ─────────────────────────────────────────────────────────────

# Country pairs that are geographically overlapping but not linked by
# P1365/P1366 in Wikidata. Keyed by frozenset of GT current countries;
# value is the set of output countries that should be flagged for the judge.
CONTESTED_COUNTRY_PAIRS = {
    frozenset({"Q801"}):     {"Q219060"},  # Israel <-> State of Palestine
    frozenset({"Q219060"}):  {"Q801"},     # reverse
    frozenset({"Q148"}):     {"Q148"},     # China (PRC) <-> Taiwan edge cases
    frozenset({"Q865"}):     {"Q148"},     # Taiwan <-> PRC
}


def evaluate_pob(gt_raw: str, output_raw: str, loq: str,
                 cache: dict, cache_path: str) -> tuple[str, float, dict]:
    """
    Evaluate a single PoB row.

    Both GT and output are resolved to candidate lists (top 5, filtered to
    geographic entities). Matching iterates over all combinations and returns
    the best score found.

    Returns (match_type, score, debug_info).
    """
    debug = {"gt_candidates": [], "out_candidates": [], "gt_countries": [], "out_countries": []}

    gt_clean  = clean_output(gt_raw)
    out_clean = clean_output(output_raw)

    if not gt_clean or not out_clean:
        return "no_match", 0.0, debug

    # ── Resolve both to candidate lists ──
    gt_candidates  = resolve_entities(gt_clean,  loq, cache, cache_path)
    out_candidates = resolve_entities(out_clean, loq, cache, cache_path)
    debug["gt_candidates"]  = gt_candidates
    debug["out_candidates"] = out_candidates

    if not gt_candidates or not out_candidates:
        return "no_match", 0.0, debug

    # ── Fetch info for all candidates ──
    gt_infos  = {q: get_entity_info(q, cache, cache_path) for q in gt_candidates}
    out_infos = {q: get_entity_info(q, cache, cache_path) for q in out_candidates}

    # ── Stage 1: exact match — any GT candidate == any output candidate ──
    if set(gt_candidates) & set(out_candidates):
        return "exact_match", SCORES["exact_match"], debug

    # ── Stage 2: alias match — labels overlap on any pair ──
    for gq in gt_candidates:
        gt_labels = set(casefold(l) for l in collect_labels(gt_infos[gq], loq))
        for oq in out_candidates:
            out_labels = set(casefold(l) for l in collect_labels(out_infos[oq], loq))
            if gt_labels & out_labels:
                return "exact_match", SCORES["exact_match"], debug

    # ── Stage 3: country-level comparison ──
    # Build full country set for each GT candidate (P17 + P1366 expansion)
    gt_current_countries  = set()
    gt_country_set        = set()  # includes historical
    for gq in gt_candidates:
        for cq in gt_infos[gq].get("countries", []):
            gt_current_countries.add(cq)
            gt_country_set.update(get_country_set(cq, cache, cache_path))
    debug["gt_countries"] = list(gt_country_set)

    # For each output candidate: determine its country
    # If candidate is itself a country, use it directly; otherwise use P17
    out_country_qs = set()
    for oq in out_candidates:
        if is_country(out_infos[oq]):
            out_country_qs.add(oq)
        else:
            out_country_qs.update(out_infos[oq].get("countries", []))
    debug["out_countries"] = list(out_country_qs)

    if not out_country_qs or not gt_country_set:
        return "no_match", 0.0, debug

    # Current country match
    if out_country_qs & gt_current_countries:
        return "country_match", SCORES["country_match"], debug

    # Historical country match
    if out_country_qs & gt_country_set:
        return "historical_match", SCORES["historical_match"], debug

    # Flag contested territories for LLM judge.
    # Certain country pairs are not linked by P1365/P1366 in Wikidata
    # (e.g. Israel / State of Palestine coexist rather than one succeeding
    # the other) but are geographically overlapping. The automatic scorer
    # cannot resolve these — route them to the judge.
    if out_country_qs & CONTESTED_COUNTRY_PAIRS.get(
        frozenset(gt_current_countries), set()
    ):
        debug["flag"] = "contested_territory"

    return "no_match", 0.0, debug


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache",  default="pob_cache.json")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows.")

    cache = load_cache(args.cache)
    print(f"Cache has {len(cache)} entries.")

    results = []
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {row['Q_number']} LoQ={row['LoQ']}")

        match_type, score, debug = evaluate_pob(
            row["gt"], row["llm_output"], row["LoQ"], cache, args.cache
        )

        results.append({
            "Q_number":      row["Q_number"],
            "LoE":           row["LoE"],
            "LoQ":           row["LoQ"],
            "property":      row["property"],
            "gt":            row["gt"],
            "llm_output":    row["llm_output"],
            "gt_candidates": json.dumps(debug["gt_candidates"]),
            "out_candidates":json.dumps(debug["out_candidates"]),
            "gt_countries":  json.dumps(debug["gt_countries"]),
            "out_countries": json.dumps(debug["out_countries"]),
            "match_type":    match_type,
            "score":         score,
            "flag":          debug.get("flag", ""),
        })

    # Write output
    fieldnames = [
        "Q_number", "LoE", "LoQ", "property", "gt", "llm_output",
        "gt_candidates", "out_candidates", "gt_countries", "out_countries",
        "match_type", "score", "flag",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    from collections import Counter
    print("\n=== Summary ===")
    match_counts = Counter(r["match_type"] for r in results)
    for mt, cnt in match_counts.most_common():
        print(f"  {mt:22s} {cnt:5d}  (score={SCORES.get(mt, 0.0)})")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
import requests, json

SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "ravenclaw-eval/0.1 (contact: andschimmenti@gmail.com)"}

queries = {
    "Q219060 P1366 (replaced by)": "SELECT ?x WHERE { wd:Q219060 wdt:P1366 ?x . }",
    "Q219060 P1365 (replaces)":    "SELECT ?x WHERE { wd:Q219060 wdt:P1365 ?x . }",
    "Q219060 P17 (country)":       "SELECT ?x WHERE { wd:Q219060 wdt:P17 ?x . }",
    "? P1366 Q219060":             "SELECT ?x WHERE { ?x wdt:P1366 wd:Q219060 . }",
    "Q801 P1365 (replaces)":       "SELECT ?x WHERE { wd:Q801 wdt:P1365 ?x . }",
}

PREFIX = "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "

for label, q in queries.items():
    resp = requests.get(SPARQL, params={"query": PREFIX + q, "format": "json"}, headers=HEADERS, timeout=30)
    results = resp.json()["results"]["bindings"]
    vals = [r["x"]["value"].split("/")[-1] for r in results]
    print(f"{label}: {vals if vals else '(empty)'}")
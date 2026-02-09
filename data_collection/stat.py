import json
import pandas as pd
from pathlib import Path

# Map each language row to: (language code, filename)
LANG_FILES = {
    "English":  ("en", "English.json"),
    "French":   ("fr", "French.json"),
    "German":   ("de", "German.json"),
    "Russian":  ("ru", "Russian.json"),
    "Italian":  ("it", "Italian.json"),
    "Arabic":   ("ar", "Arabic.json"),
    "Polish":   ("pl", "Polish.json"),
    "Chinese":  ("zh", "Chinese.json"),
    "Hindi":    ("hi", "Hindi.json"),
}

ALL_LANG_CODES = [lang_code for _, (lang_code, _) in LANG_FILES.items()]

PROPS = ["P20", "P19", "P106", "P27"]

def has_entity_label_in_lang(entity: dict, lang: str) -> bool:
    return lang in (entity.get("labels") or {})

def entity_is_complete(entity: dict) -> bool:
    # entity is "complete" iff all 4 props have at least one value labeled in ALL languages
    return all(prop_has_value_labeled_in_all_lang(entity, p) for p in PROPS)


def prop_has_value_labeled_in_lang(entity: dict, prop: str, lang: str) -> bool:
    """
    True iff entity has property `prop` and at least one of its values has a label in `lang`.
    """
    p = entity.get(prop)
    if not p:
        return False
    values = p.get("values") or []
    if not values:
        return False
    return any(lang in (v.get("labels") or {}) for v in values)

def has_labels_in_all_langs(entity: dict) -> bool:
    labels = entity.get("labels") or {}
    return all(code in labels for code in ALL_LANG_CODES)

def prop_has_value_labeled_in_all_lang(entity: dict, prop: str) -> bool:
    """
    True iff entity has property `prop` and one of its values have labels in all languages.
    """
    p = entity.get(prop)
    if not p:
        return False
    values = p.get("values") or []
    if not values:
        return False
    return any(has_labels_in_all_langs(v) for v in values)

def compute_row_for_file(json_path: Path, lang_code: str, sitelinks_min: int = 9) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # your JSON is a dict keyed by QID -> entity dict
    entities = list(data.values())

    row = {"#Entities": len(entities)}

    # filter by sitelinks >= 9
    E = [e for e in entities if int(e.get("sitelinks", 0)) >= sitelinks_min]

    # per-property counts
    for prop in PROPS:
        row[prop] = sum(prop_has_value_labeled_in_all_lang(e, prop) for e in E)

    # complete(L) = #entities with labels in all languages and values for all PROPS labeled in L
    row["complete"] = sum(
        all(prop_has_value_labeled_in_all_lang(e, p) for p in PROPS)
        for e in E
    )
    return row

def build_table(folder: str, sitelinks_min: int = 9) -> pd.DataFrame:
    folder = Path(folder)
    rows = []
    for lang_name, (lang_code, filename) in LANG_FILES.items():
        row = {"Language": lang_name}
        row.update(compute_row_for_file(folder / filename, lang_code, sitelinks_min=sitelinks_min))
        rows.append(row)

    return pd.DataFrame(rows, columns=["Language", "#Entities", "complete", *PROPS])

def write_complete_jsons(input_folder: str, output_folder: str, sitelinks_min: int = 9) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for lang_name, (lang_code, filename) in LANG_FILES.items():
        in_path = input_folder / filename
        out_path = output_folder / f"{lang_name}_complete.json"

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # dict: qid -> entity

        filtered = {
            qid: ent
            for qid, ent in data.items()
            if int(ent.get("sitelinks", 0)) >= sitelinks_min and entity_is_complete(ent)
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        print(f"{lang_name}: wrote {len(filtered)} entities -> {out_path}")


# Example usage:
df = build_table("data/dataset_v2/entities_data_complete/", sitelinks_min=9)
print(df.to_string(index=False))
df.to_csv("data/dataset_v2/statistics/stats_complete.csv", index=False)

write_complete_jsons(
    input_folder="data/dataset_v2/entities_data/",
    output_folder="data/dataset_v2/entities_data_complete/",
    sitelinks_min=9
)

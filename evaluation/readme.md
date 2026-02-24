# Evaluation Scripts Documentation

This directory contains specialized evaluation scripts for assessing LLM performance on different Wikidata properties. Currently in development, it misses the LLM-as-a-judge approach to be implemented in all the scripts.
- [x] evaluate_country.py
- [x] evaluate_dob.py
- [x] evaluate_pob.py
- [x] occupation/score.py


## Available Scripts

### 1. **evaluate_country.py** - Country of Citizenship Evaluation
**Purpose**: Evaluates LLM performance on property P27 (country of citizenship)

**Usage**:
```bash
python evaluate_country.py --input scored.csv --output results.csv --cache vocab_cache.json
```

**Parameters**:
- `--input`: Path to scored CSV (must contain columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
- `--output`: Path for results CSV
- `--cache`: Path to vocabulary cache JSON (created if missing, updated on each new country fetch)

**Scoring System**:
- `exact_match`: 1.0 - Direct country name match
- `alias_match`: 0.9 - Match with country alias/alternative name
- `substring_match`: 0.85 - Partial match within country name
- `demonym_match`: 0.8 - Match with demonym (e.g., "American" for "United States")
- `no_match`: 0.0 - No match found

**Features**:
- Queries Wikidata SPARQL endpoint for country information
- Supports multiple target languages (en, fr, de, it, ru, pl, ar, zh, hi)
- Caches vocabulary data to improve performance
- Handles various country name formats and demonym matching

### 2. **evaluate_dob.py** - Date of Birth Evaluation
**Purpose**: Evaluates LLM performance on date of birth extraction and formatting

**Usage**:
```bash
python evaluate_dob.py --input scored.csv --output results_dob.csv
```

**Parameters**:
- `--input`: Path to scored CSV (columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
- `--output`: Path for results CSV

**Scoring System**:
- `exact_match`: 1.0 - YYYY-MM-DD format matches directly
- `swap_match`: 1.0 - Match after swapping month and day in output
- `no_match`: 0.0 - No valid date match

**Features**:
- Handles both ISO format with timestamp (2024-01-03T00:00:00Z) and plain format (2024-01-03)
- Robust date parsing with regex validation
- Accounts for common day/month swap errors
- No external API dependencies

### 3. **evaluate_pob.py** - Place of Birth Evaluation
**Purpose**: Evaluates LLM performance on property P19 (place of birth)

**Usage**:
```bash
python evaluate_pob.py --input scored.csv --output results_pob.csv --cache pob_cache.json
```

**Parameters**:
- `--input`: Path to scored CSV (columns: Q_number, LoE, LoQ, property, gt, llm_output, prompt)
- `--output`: Path for results CSV
- `--cache`: Path to cache JSON (created if missing, updated on each fetch)

**Scoring System**:
- `exact_match`: 1.0 - Exact place name match
- `country_match`: 0.8 - Match at country level only
- `historical_match`: 0.7 - Match with historical place names
- `no_match`: 0.0 - No match found

**Features**:
- Queries both Wikidata SPARQL endpoint and Wikibase API
- Handles historical place names and administrative changes
- Multi-language support for place names
- Comprehensive caching system for place data
- Geographic hierarchy matching (city → region → country)

## Common Input Format

All scripts expect input CSV files with the following columns:

- `Q_number`: Wikidata entity identifier
- `LoE`: Level of English
- `LoQ`: Level of Question
- `property`: Wikidata property being evaluated
- `gt`: Ground truth value
- `llm_output`: LLM-generated response
- `prompt`: Original prompt used

## Output Format

All scripts generate CSV files with additional columns:

- `match_type`: Type of match achieved
- `score`: Numerical score (0.0-1.0)
- Original input columns preserved

## Running Evaluations

### Country of Citizenship (P27)
```bash
python evaluate_country.py --input test_country_eval.csv --output country_results.csv --cache vocab_cache.json
```

### Date of Birth
```bash
python evaluate_dob.py --input scored.csv --output dob_results.csv
```

### Place of Birth (P19)
```bash
python evaluate_pob.py --input test_pob_eval.csv --output pob_results.csv --cache pob_cache.json
```

### 4. **occupation/score.py** - Occupation Scoring (NLP Metrics)
**Purpose**: Scores occupation predictions using standard NLP metrics (BERTScore, BLEURT, XCOMET, NLGEval)

**Usage**:
```bash
python occupation/score.py --input /path/to/csvs --output /path/to/output [--metric bert|bleurt|xcomet|nlgeval]
```

**Parameters**:
- `--input`: Path to the folder containing input CSVs (matched by `property_pob_*.csv` pattern)
- `--output`: Path to the folder where scored CSVs will be saved (created if missing)
- `--metric`: Scoring metric to use (default: `bert`)

**Available Metrics**:
- `bert`: BERTScore F1 using `bert-base-multilingual-cased` — output column `BERT_score`
- `bleurt`: BLEURT score — output column `BERT_score`
- `xcomet`: XCOMET-XXL score — output column `xCOMET`
- `nlgeval`: BLEU 1–4, CIDEr, METEOR, ROUGE-L — output columns `Bleu_1` … `ROUGE_L`

**Notes**:
- No API key required — this script uses local/downloaded models only
- `xcomet` requires a GPU (`gpus=1`)
- Output CSVs are prefixed with the metric name (e.g., `bertscore_<filename>.csv`)

---

## Notes

- Scripts with cache parameters will create cache files on first run and reuse them for subsequent runs
- Cache files improve performance by storing previously fetched Wikidata data
- All scripts handle various input formats and provide robust error handling
- Results include detailed match types for analysis of LLM performance patterns
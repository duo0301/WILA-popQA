# Submission for LREC 2026

This repository contains the code and multilingual dataset for our LREC 2026 submission "A Wikidata-Based Framework to Measure Cross-Lingual Bias in Multilingual Large Language Models." The interdisciplinary project is focused on understanding cultural biases in multilingual LLMs. It explores how language alignment affects the performance of LLMs, particularly when answering factual questions about cultural entities such as authors, historical figures, and traditional practices. By leveraging Wikidata Knowledge Graphs, this project aims to provide a thorough analysis of the capabilities and limitations of multilingual LLMs.

## Directory Structure

- **`data_collection/`** — scripts to retrieve and process Wikidata entities and generate prompts
  - **`data/`** — intermediate and processed dataset files
- **`evaluation/`**
  - **`evaluation_scripts/`** — evaluation scripts, organized per property
  - **`reports/`** — report generation scripts and output reports
- **`inference/`** — scripts for running batch inference on LLMs

## Installation and Setup

Requires Python 3.11 or higher and [uv](https://docs.astral.sh/uv/).

To install dependencies run: 

```bash
uv sync
```

## Data Access

Large data files (experiments, evaluations, prompts) are versioned with [DVC](https://dvc.org) and stored on Google Drive. To download them:

1. **Configure the DVC remote**:   

```bash                                     
   uv run dvc remote modify --local gdrive gdrive_service_account_json_file_path /path/to/service_account_key.json                           
```                                         

2. **Pull the data**:

```bash                                     
uv run dvc pull                             
``` 

To reproduce the full pipeline:
```bash
uv run dvc repro
```

Individual stages can be run with:
```bash
uv run dvc repro <stage_name>
```

To visualize the dependency graph run: 

```bash
uv run dvc dag
```

See `dvc.yaml` for the full pipeline definition and available stages.


## License
This project is licensed under the MIT License. Refer to the `LICENSE` file for more details.

## Contact
If you have any questions, feel free to contact us via GitHub or reach out to the contributors directly.

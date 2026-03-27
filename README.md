# WILA-popQA

This repository contains the code and multilingual dataset for our LREC 2026 workshop paper "A Wikidata-Based Framework to Measure Cross-Lingual Bias in Multilingual Large Language Models." The interdisciplinary project is focused on understanding cultural biases in multilingual LLMs. It explores how language alignment affects the performance of LLMs, particularly when answering factual questions about cultural entities such as authors, historical figures, and traditional practices. By leveraging Wikidata Knowledge Graphs, this project aims to provide a thorough analysis of the capabilities and limitations of multilingual LLMs.

## Directory Structure

- **`data_collection/`** — scripts to retrieve and process Wikidata entities and generate prompts
  - **`data/`** — intermediate and processed dataset files
- **`evaluation/`**
  - **`evaluation_scripts/`** — evaluation scripts, organized per property
  - **`reports/`** — report generation scripts and output reports
- **`inference/`** — scripts for running batch inference on LLMs

## Installation and Setup

Requires Python 3.11 or higher, [uv](https://docs.astral.sh/uv/) and [Docker](https://www.docker.com/). Large data files (experiments, evaluations, prompts) are versioned with [DVC](https://dvc.org) and stored on Google Drive. A public version of our data is available [here](https://zenodo.org/records/19249706) and is required to reproduce the pipeline.

1. Clone the repository.

2. To install dependencies run: 

```bash
uv sync
```

3. Download our data via [this link](https://file.fast/NVZ63/ext-data.zip). The data includes the processed dataset, prompts, and evaluation results.

4. Unzip the downloaded folder in the root directory of the cloned repository 

2. (optional) Follow the steps outlined in the [DVC documentation](https://doc.dvc.org/user-guide/data-management/remote-storage) to set up your own DVC remote.

4. (optional) To push the files to your DVC remote: 

```bash
uv run dvc push
```

## Usage 

To reproduce the full pipeline, run:

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=6G qacompany/qendpoint-wikidata
uv run dvc repro
```

Individual stages can be run with:

```bash
uv run dvc repro <stage_name>
```
Please note that the stages `get_entities` and `get_data` require running the docker command above. 

To visualize the dependency graph run: 

```bash
uv run dvc dag
```

See `dvc.yaml` for the full pipeline definition and available stages.


## For Contributors 

To download the latest file versions and run the pipeline with the Google Drive remote storage, access credentials as well as the Google folder ID are required. Configure the DVC remote with: 

```bash
uv run dvc remote modify --local gdrive url gdrive://<FOLDER_ID>
uv run dvc remote modify --local gdrive gdrive_client_id 'client-id'
uv run dvc remote modify --local gdrive gdrive_client_secret 'client-secret'
                       
```
To pull the latest data from the remote, run: 

```bash                                     
uv run dvc pull                             
``` 

## License
This project is licensed under the MIT License. Refer to the `LICENSE` file for more details.

## Contact
If you have any questions, feel free to contact us via GitHub or reach out to the contributors directly.

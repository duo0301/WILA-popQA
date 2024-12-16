# Ravenclaw Project

Welcome to the Ravenclaw Project repository!
The Ravenclaw Project is an interdisciplinary research initiative focused on understanding cultural biases in multilingual LLMs. It explores how language alignment affects the performance of LLMs, particularly when answering factual questions about cultural entities such as authors, historical figures, and traditional practices. By leveraging Wikidata Knowledge Graphs, this project aims to provide a thorough analysis of the capabilities and limitations of multilingual LLMs.

## Directory Structure
This repository contains the following main directories, each playing a specific role in the workflow:

- **`data_collection/`**: Scripts and notebooks to retrieve and process data from Wikidata, focusing on cultural entities using tools like Pywikibot and Blazegraph,qEndPoint.
- **`evaluation/`**: Notebooks for evaluating QA results, analyzing metrics such as accuracy, precision, recall, and F1 scores.
- **`inference/`**: Scripts for determining the Language of the Entity (LoE) and analyzing multilingual aspects of Wikidata entities.
- **`popularity_metric/`**: Scripts and documentation for calculating popularity metrics like Wikipedia sitelinks and analyzing their influence.
- **`popularity_normalisation/`**: Resources related to normalizing popularity metrics, including QRank calculations.
- **`prompt_construction/`**: Templates and methods for constructing multilingual prompts during the QA process.

## Installation and Setup
To set up the Ravenclaw Project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adrian9631/ravenclaw_project.git
   cd ravenclaw_project
   ```

2. **Install dependencies**:
   - Ensure you have Python 3.7 or higher.
   - Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
### Data Collection
To retrieve cultural entity data from Wikidata, use the scripts provided in the `data_collection/` directory. These scripts help gather data that serves as the foundation for analyzing LLMs.

### Running Evaluation
Once the dataset is complete, navigate to the `evaluation/` folder to assess the QA performance using metrics such as accuracy, precision, recall, and F1 scores.

## Contributors
The Ravenclaw Project is a collaborative effort involving contributions from:

- [Adrian9631 (Duo)](https://github.com/adrian9631)
- [Lipogg (Lisa Poggel)](https://github.com/lipogg)
- [Miferroudjene (Mouloud Iferroudjene)](https://github.com/miferroudjene)
- [Aschimmenti (Andrea Schimmenti)](https://github.com/aschimmenti)
- [Marta Boscariol](https://github.com/martaboscariol)
- [Jan Kalo](https://github.com/JanKalo)
- [Kanchanks (Kanchan Shivashankar)](https://github.com/kanchanks)

## How to Contribute
We welcome contributions to make the Ravenclaw Project better. Here’s how you can help:

- **Improve Data Retrieval**: Enhance efficiency and reliability of data retrieval scripts.
- **Add New Popularity Metrics**: Extend the scope of popularity metrics to include more diverse measures.
- **Extend QA Templates**: Create QA templates for additional languages or expand them to cover new entity classes.

To contribute, fork the repository and submit a pull request. Refer to our [issues page](https://github.com/adrian9631/ravenclaw_project/issues) for open tasks that need attention.

## License
This project is licensed under the MIT License. Refer to the `LICENSE` file for more details.

## Contact
If you have any questions, feel free to contact us via GitHub or reach out to the contributors directly.

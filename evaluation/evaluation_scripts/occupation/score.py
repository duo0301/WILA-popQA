import argparse
import pprint
import pandas as pd
import os
import glob
from tqdm import tqdm
tqdm.pandas()

def BERT_score(input_folder: str, output_folder: str):
    """
    Reads CSVs from input_folder, computes scores based on two columns,
    adds new columns, and saves updated CSVs into output_folder.

    Args:
        input_folder (str): Path to the folder containing original CSVs.
        output_folder (str): Path to the folder where modified CSVs will be saved.
    """

    from bert_score import BERTScorer
    # https://github.com/Tiiiger/bert_score
    # issues: (solution: rescaling)
    # https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md 
    # https://github.com/Tiiiger/bert_score/issues/44

    # model choices: https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py
    # model_type = 'microsoft/mdeberta-v3-base'
    # model_type = 'google/mt5-large'
    model_type = 'bert-base-multilingual-cased'
    scorer = BERTScorer(model_type=model_type, nthreads=4, batch_size=1024, idf=False, rescale_with_baseline=False)

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    pattern = os.path.join(input_folder, "property_pob_*.csv")
    # pattern = os.path.join(input_folder, "property_*_*.csv")
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Example: Assuming we score based on 'gt' and 'llm_output' columns
        if 'gt' in df.columns and 'llm_output' in df.columns:
            P, R, F1 = scorer.score(df['gt'].tolist(), df['llm_output'].tolist())
            # Add scores back to DataFrame
            df['BERT_score'] = F1
        else:
            print(f"Warning: 'gt' or 'llm_output' columns missing in {filename}. Skipping scoring.")

        # Save the updated DataFrame
        output_path = os.path.join(output_folder, 'bertscore_'+ filename)
        df = df.drop(columns=["prompt"])
        df.to_csv(output_path, index=False)
        print(f"Saved updated file to {output_path}")

def BLEURT_score(input_folder: str, output_folder: str):
    """
    Reads CSVs from input_folder, computes scores based on two columns,
    adds new columns, and saves updated CSVs into output_folder.

    Args:
        input_folder (str): Path to the folder containing original CSVs.
        output_folder (str): Path to the folder where modified CSVs will be saved.
    """

    from bleurt import score
    # https://github.com/google-research/bleurt

    # checkpoint = "bleurt/test_checkpoint"
    scorer = score.BleurtScorer()

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    pattern = os.path.join(input_folder, "property_pob_*.csv")
    # pattern = os.path.join(input_folder, "property_*_*.csv")
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Example: Assuming we score based on 'gt' and 'llm_output' columns
        if 'gt' in df.columns and 'llm_output' in df.columns:
            # P, R, F1 = scorer.score(df['gt'].tolist(), df['llm_output'].tolist(), batch_size=1024)
            # Add scores to DataFrame
            df['BERT_score'] = scorer.score(references=df['gt'].tolist(), candidates=df['llm_output'].tolist(), batch_size=1024)
        else:
            print(f"Warning: 'gt' or 'llm_output' columns missing in {filename}. Skipping scoring.")

        # Save the updated DataFrame
        output_path = os.path.join(output_folder, 'bleurt_'+ filename)
        df = df.drop(columns=["prompt"])
        df.to_csv(output_path, index=False)
        print(f"Saved updated file to {output_path}")

def XCOMET_score(input_folder: str, output_folder: str):
    """
    Reads CSVs from input_folder, computes scores based on two columns,
    adds new columns, and saves updated CSVs into output_folder.

    Args:
        input_folder (str): Path to the folder containing original CSVs.
        output_folder (str): Path to the folder where modified CSVs will be saved.
    """

    from comet import download_model, load_from_checkpoint
    # https://github.com/Unbabel/COMET
    # https://huggingface.co/Unbabel/XCOMET-XL

    model_path = download_model("Unbabel/XCOMET-XXL")
    model = load_from_checkpoint(model_path)

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    pattern = os.path.join(input_folder, "property_pob_*.csv")
    # pattern = os.path.join(input_folder, "property_*_*.csv")
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Example: Assuming we score based on 'gt' and 'llm_output' columns
        if 'gt' in df.columns and 'llm_output' in df.columns:

            # Suppose df has columns: 'gt', 'llm_output'
            batch_size = 256

            # Prepare list of input dicts
            inputs = [{'src': row['gt'], 'mt': row['llm_output']} for _, row in df.iterrows()]

            # Split inputs into batches
            def batchify(lst, batch_size):
                for i in range(0, len(lst), batch_size):
                    yield lst[i:i + batch_size]

            # Run model in batches
            all_scores = []
            for batch in batchify(inputs, batch_size):
                model_output = model.predict(batch, batch_size=batch_size, gpus=1)  # get list of scores
                all_scores.extend(model_output.scores)  # keep results in order

            # Add scores back to DataFrame
            df['xCOMET'] = all_scores
        else:
            print(f"Warning: 'gt' or 'llm_output' columns missing in {filename}. Skipping scoring.")

        # Save the updated DataFrame
        output_path = os.path.join(output_folder, 'xcomet_'+ filename)
        df = df.drop(columns=["prompt"])
        df.to_csv(output_path, index=False)
        print(f"Saved updated file to {output_path}")

def NLGEval_score(input_folder: str, output_folder: str):
    """
    Reads CSVs from input_folder, computes scores based on two columns,
    adds new columns, and saves updated CSVs into output_folder.

    Args:
        input_folder (str): Path to the folder containing original CSVs.
        output_folder (str): Path to the folder where modified CSVs will be saved.
    """

    from nlgeval import NLGEval
    # https://github.com/Maluuba/nlg-eval

    nlgeval = NLGEval(
        no_skipthoughts=True,
        no_glove=True,
        metrics_to_omit=[
            'SPICE',
            'EmbeddingAverageCosineSimilarity',
        ]
    )  # loads the models

    def ngl_eval(gt, pred):
        metrics = nlgeval.compute_individual_metrics(gt, pred)
        return metrics['Bleu_1'], metrics['Bleu_2'], metrics['Bleu_3'], metrics['Bleu_4'], metrics['CIDEr'], metrics[
            'METEOR'], metrics['ROUGE_L']

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    pattern = os.path.join(input_folder, "property_pob_*.csv")
    # pattern = os.path.join(input_folder, "property_*_*.csv")
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Example: Assuming we score based on 'gt' and 'llm_output' columns
        if 'gt' in df.columns and 'llm_output' in df.columns:
            df[['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']] = df.progress_apply(
                lambda row: pd.Series(ngl_eval(row['gt'], row['llm_output'])), axis=1
            )
        else:
            print(f"Warning: 'gt' or 'llm_output' columns missing in {filename}. Skipping scoring.")

        # Save the updated DataFrame
        output_path = os.path.join(output_folder, 'nlgeval_'+filename)
        df = df.drop(columns=["prompt"])
        df.to_csv(output_path, index=False)
        print(f"Saved updated file to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Score occupation predictions using various NLP metrics.")
    parser.add_argument("--input", required=True, help="Path to the folder containing input CSVs.")
    parser.add_argument("--output", required=True, help="Path to the folder where scored CSVs will be saved.")
    parser.add_argument(
        "--metric",
        choices=["bert", "bleurt", "xcomet", "nlgeval"],
        default="bert",
        help="Scoring metric to use (default: bert).",
    )
    args = parser.parse_args()

    if args.metric == "bert":
        BERT_score(args.input, args.output)
    elif args.metric == "bleurt":
        BLEURT_score(args.input, args.output)
    elif args.metric == "xcomet":
        XCOMET_score(args.input, args.output)
    elif args.metric == "nlgeval":
        NLGEval_score(args.input, args.output)


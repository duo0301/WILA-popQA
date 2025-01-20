import json
import pprint
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from pt_dataset import MultilingualQADataset_v1, MultilingualQADataset_v2, collate_fn

# fp8 inference settings
# Llama-3.1-70B
# import transformer_engine.pytorch as te

# llm configuration
# max_new_tokens - control the maximum numbers of tokens to generate
def get_generation_args(model_id, tokenizer):

    if model_id == "meta-llama/Meta-Llama-3-8B-Instruct" or \
            model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or \
            model_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generation_args = {
            "eos_token_id":terminators,
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
        }

    elif model_id == "mistralai/Mistral-7B-Instruct-v0.2":

        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.1,
        }

    elif model_id in ["google/gemma-1.1-7b-it", "google/gemma-2-9b-it"]:

        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.1,
        }

    elif model_id in ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct"]:
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.1,
        }
    else:
        raise ValueError("Could not find the provided model id. ")

    return generation_args

def inference(model_id, dataset):

    # load pytorch dataset
    dataset._set_model_id(model_id)

    print("Total questions: ", len(dataset))
    print("Example: ")
    pprint.pprint (dataset.__getitem__(50))

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    generation_args = get_generation_args(model_id, tokenizer)
    if model_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        # torch.backends.cuda.matmul.allow_fp8 = True
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if model_id in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        batch_size = 256
    elif model_id in [
        "google/gemma-2-9b-it",
    ]:
        batch_size = 16
    elif model_id in [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ]:
        batch_size = 4
    else:
        batch_size = 16

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)

    all_Q_number = [] # q number (entity)
    all_LoE = [] # language of entity
    all_LoQ = [] # language of question
    all_property = [] # property (question)
    all_gt = [] # ground truth
    all_prompt = [] # prompt
    all_output = [] # llm_output

    for batch in dataloader:

        all_Q_number.extend(batch['Q_number'])
        all_LoE.extend(batch['LoE'])
        all_LoQ.extend(batch['LoQ'])
        all_property.extend(batch['property'])
        all_prompt.extend(batch['prompt'])
        all_gt.extend(batch['gt'])

        texts = tokenizer.apply_chat_template(batch['input'], add_generation_prompt=False, tokenize=False)
        inputs = tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.cuda() for key, val in inputs.items()}
        previous_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        gen_tokens = model.generate(
            **inputs,
            **generation_args
        )
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        if 'meta-llama' in model_id:
            # remove prefix, e.g., 'assistant\n\nМарокканское.'
            gen_text = [full_text[len(previous_texts[idx]) + 11:] for idx, full_text in enumerate(gen_text)]
        else:
            gen_text = [full_text[len(previous_texts[idx]):] for idx, full_text in enumerate(gen_text)]
        all_output.extend(gen_text)
        pprint.pprint (gen_text)

    df = pd.DataFrame({
        "Q_number": all_Q_number,
        "LoE": all_LoE,
        "LoQ": all_LoQ,
        "property": all_property,
        "gt": all_gt,
        "llm_output": all_output,
        "prompt": [prompt.replace("\n", " ") for prompt in all_prompt]
    })

    # Get unique values from the 'property' column
    unique_properties = df['property'].unique()

    # Split and save each group as a separate CSV
    for prop in unique_properties:
        subset_df = df[df['property'] == prop]  # Filter rows for this property
        filename = f"property_{prop}_{model_id.split('/')[0]}.csv"  # Create filename dynamically
        subset_df.to_csv(filename, index=False)
        print(f"Saved rows with property '{prop}' to {filename}")

def main():

    # data_dir = '/apollo/dya/ISWS/data_v1/Prompt_final'
    data_dir = '/apollo/dya/ISWS/data_v2'
    lang_set = ['ar', 'en', 'de', 'fr', 'it', 'pl', 'ru', 'zh']
    properties = ['pob', 'dob', 'occ', 'country']
    dataset = MultilingualQADataset_v2(data_dir, lang_set, properties)

    model_ids = [
        # "microsoft/Phi-3-mini-4k-instruct", # not working well
        # "microsoft/Phi-3.5-mini-instruct", # not working well
        "mistralai/Mistral-7B-Instruct-v0.2", # good
        # "meta-llama/Meta-Llama-3-8B-Instruct", # good
        # "meta-llama/Meta-Llama-3.1-8B-Instruct", # good
        # "meta-llama/Meta-Llama-3.1-70B-Instruct", # good
        # "google/gemma-1.1-7b-it", # good
        # "google/gemma-2-9b-it" # not working well
    ]

    num_run = 1
    for model_id in model_ids:
        for run_id in range(num_run):
            print ( "model {} - run {}".format(model_id, run_id) )
            inference(model_id, dataset)

if __name__ == '__main__':
    main()
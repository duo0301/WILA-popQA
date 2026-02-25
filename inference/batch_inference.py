import json
import pprint
import pandas as pd

import re

import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from pt_dataset import MultilingualQADataset, collate_fn

def get_generation_args(model_id, tokenizer):

    if model_id in ["meta-llama/Meta-Llama-3.1-8B-Instruct"]:

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generation_args = {
            "eos_token_id":terminators,
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        }

    elif model_id in ["mistralai/Mistral-7B-Instruct-v0.3"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["google/gemma-2-9b-it", "google/gemma-3-12b-it"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["zai-org/glm-4-9b-chat-hf"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["allenai/Olmo-3-7B-Instruct"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["nvidia/NVIDIA-Nemotron-Nano-9B-v2"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["moonshotai/Moonlight-16B-A3B-Instruct"]:

        generation_args = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }

    elif model_id in ["microsoft/phi-4"]:
        generation_args = {
            # https://huggingface.co/microsoft/phi-4/discussions/38
            "eos_token_id": [100257, 100265],
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.6,
        }
    elif model_id in ["deepseek-ai/DeepSeek-V2-Lite-Chat"]:
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.6,
        }
    else:
        raise ValueError("Could not find the provided model id. ")

    return generation_args

def inference(model_id, dataset, output_dir):


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        # https://github.com/vllm-project/vllm/issues/6177
        torch_dtype=torch.bfloat16,
        # https://github.com/huggingface/transformers/issues/32848
        # attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    generation_args = get_generation_args(model_id, tokenizer)

    if model_id in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "zai-org/glm-4-9b-chat-hf",
        "google/gemma-3-12b-it",
        "google/gemma-2-9b-it",
        "Qwen/Qwen3-8B",
    ]:
        batch_size = 256
        
    elif model_id in [
        "allenai/Olmo-3-7B-Instruct",
        "microsoft/phi-4",
        "Qwen/Qwen3-14B",
    ]:
        batch_size = 128

    elif model_id in [
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "moonshotai/Moonlight-16B-A3B-Instruct",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    ]:
        batch_size = 64
    
    else:
        batch_size = 32

    # 2) load pytorch dataset and data loader
    dataset._set_model_id(model_id)

    print (model_id)

    print("Total questions: ", len(dataset))
    print("Example: ")
    pprint.pprint (dataset.__getitem__(50))

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)

    all_Q_number = [] # q identifier (entity)
    all_LoE = [] # language of entity
    all_LoQ = [] # language of question
    all_property = [] # property (question)
    all_gt = [] # ground truth
    all_prompt = [] # prompt
    all_output = [] # llm_output

    # 3) inference
    for batch in dataloader:

        all_Q_number.extend(batch['Q_number'])
        all_LoE.extend(batch['LoE'])
        all_LoQ.extend(batch['LoQ'])
        all_property.extend(batch['property'])
        all_prompt.extend(batch['prompt'])
        all_gt.extend(batch['gt'])

        if model_id in [ "Qwen/Qwen3-8B", "Qwen/Qwen3-14B" ]:
            texts = tokenizer.apply_chat_template(batch['input'], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        else:
            texts = tokenizer.apply_chat_template(batch['input'], add_generation_prompt=True, tokenize=False)
            
        inputs = tokenizer(texts, padding="longest", return_tensors="pt").to(model.device)
        previous_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        gen_tokens = model.generate(
            **inputs,
            **generation_args
        )
        # saving raw generated outputs
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        gen_text = [full_text[len(previous_texts[idx]):].strip() for idx, full_text in enumerate(gen_text)]

        # post-processing only for deepseek
        # removing the thinking part based on '</think>'
        if "deepseek-ai/DeepSeek-R1" in model_id:
            deepseek_gen_text = []
            for _text in gen_text:
                if "</think>" in _text:
                    deepseek_gen_text.append(_text.split('</think>')[1].strip())
                # The thinking process exceeded the max_new_tokens limit and did not include with the end symbol </think>.
                else:
                    deepseek_gen_text.append(_text.split('\n')[-1].strip())
            gen_text = deepseek_gen_text

        pprint.pprint(gen_text)
        all_output.extend(gen_text)

    # 4) save the results
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
        subset_df = df[df['property'] == prop]  
        filename = f"property_{prop}_{model_id.split('/')[1]}.csv"  
        file_path = output_dir + filename
        subset_df.to_csv(file_path , index=False)
        print(f"Saved rows with property '{prop}' to {file_path}")

def main():

    parser = argparse.ArgumentParser(description="Multilingual QA Inference")
    parser.add_argument("--batch_dir", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save results")
    args = parser.parse_args()

    data_dir = args.batch_dir
    output_dir = args.output_dir

    lang_set = ['ar', 'en', 'de', 'fr', 'it', 'pl', 'ru', 'zh', 'hi']
    properties = ['pob', 'dob', 'occ', 'country']
    dataset = MultilingualQADataset(data_dir, lang_set, properties)

    model_ids = [
        "microsoft/phi-4",                            # 15B
        "mistralai/Mistral-7B-Instruct-v0.3",         # 7B
        "meta-llama/Meta-Llama-3.1-8B-Instruct",      # 8B
        "google/gemma-2-9b-it",                       # 9B
        "google/gemma-3-12b-it",                      # 12B
        "Qwen/Qwen3-8B"                               # 8B
        "Qwen/Qwen3-14B"                              # 14B
        "zai-org/glm-4-9b-chat-hf"                    # 9B
        "allenai/Olmo-3-7B-Instruct"                  # 7B
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2"           # 9B
        "moonshotai/Moonlight-16B-A3B-Instruct"       # 16B A3B
        "deepseek-ai/DeepSeek-V2-Lite-Chat"           # 16B A3B
    ]

    num_run = 1
    for model_id in model_ids:
        for run_id in range(num_run):
            print ( "model {} - run {}".format(model_id, run_id) )
            inference(model_id, dataset, output_dir)

if __name__ == '__main__':
    main()
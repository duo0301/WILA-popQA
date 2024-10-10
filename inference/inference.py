import torch
import json
import argparse
from ast import literal_eval
from functions import get_messages, \
    fetch_data_single_prompt_multiple_questions, \
    save_data_single_prompt_multiple_questions
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# llm configuration
def get_generation_args(model_id, tokenizer):

    if model_id == "meta-llama/Meta-Llama-3-8B-Instruct" or model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generation_args = {
            "eos_token_id":terminators,
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "return_full_text": False
        }

    elif model_id == "mistralai/Mistral-7B-Instruct-v0.2":

        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "return_full_text": False
        }

    elif model_id == "google/gemma-1.1-7b-it" or model_id == "google/gemma-2-9b-it":

        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "return_full_text": False
        }

    elif model_id == "microsoft/Phi-3-mini-4k-instruct":
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.0,
            "return_full_text": False
        }
    else:
        raise ValueError("Could not find the provided model id. ")

    return generation_args

def llm_inference(model_id, num_run):

    # settings
    device = "cuda:0"
    dtype = torch.bfloat16

    # model pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    for run_id in range(num_run):
        for lang in ['en', 'de', 'it', 'pl', 'ru']:
            for lang2 in ['en', 'de', 'it', 'pl', 'ru']:
                for en_instruction, prompt, fname in fetch_data_single_prompt_multiple_questions(lang, lang2):

                    # sometimes it pops up JSON parsing error because llm can not output in correct format, e.g., punctuation missing
                    tmp_dict = None
                    while True:
                        try:
                            messages = get_messages(model_id, en_instruction, prompt)
                            generation_args = get_generation_args(model_id, tokenizer)
                            output = pipe(messages, **generation_args)
                            if generation_args["return_full_text"]:
                                returned_text =  (output[0]["generated_text"][-1]['content'])
                            else:
                                returned_text =  output[0]['generated_text']

                            text_output = returned_text.strip()
                            print("---------------------text_output-----------------")
                            print(text_output)
                            print("-------------------------------------------------")
                            dict_output = text_output.split('}')[0] + '}'
                            tmp_dict = literal_eval(dict_output)

                            save_data_single_prompt_multiple_questions(model_id, returned_text, fname, lang, lang2, run_id)
                        except:
                            pass

                        if tmp_dict is not None:
                            break

def main():

    # For a question list regarding one entity, we gather 3 outputs from each llm
    num_run = 3
    for model_id in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-1.1-7b-it",
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # "google/gemma-2-9b-it"
    ]:
        llm_inference(model_id, num_run)

if __name__ == "__main__":
    main()

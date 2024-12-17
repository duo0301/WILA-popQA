import torch
import json
import argparse
from ast import literal_eval
from functions import get_messages, \
    fetch_data_single_prompt_multiple_questions, \
    save_data_single_prompt_multiple_questions
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# fp8 inference settings
# import transformer_engine.pytorch as te

# llm configuration
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

    elif model_id == "microsoft/Phi-3-mini-4k-instruct" or model_id == "microsoft/Phi-3.5-mini-instruct":
        generation_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.6,
            "return_full_text": False
        }
    else:
        raise ValueError("Could not find the provided model id. ")

    return generation_args

def llm_inference(model_id, num_run):

    # model pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if model_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":

        # torch.backends.cuda.matmul.allow_fp8 = True

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0",
            load_in_4bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.float32,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    for run_id in range(num_run):
        # entity language
        for lang in ['en', 'de', 'it', 'pl', 'ru']:
            # prompt language
            for lang2 in ['en', 'de', 'it', 'pl', 'ru']:
                for en_instruction, prompt, fname in fetch_data_single_prompt_multiple_questions(lang, lang2):

                    # sometimes it pops up JSON parsing error because llm can not output in correct format, e.g., punctuation missing
                    tmp_dict = None

                    messages = get_messages(model_id, en_instruction, prompt)
                    generation_args = get_generation_args(model_id, tokenizer)

                    while True:

                        # FP8 inference
                        # if model_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":
                        #     with torch.cuda.amp.autocast(dtype=torch.float8):
                        #         output = pipe(messages, **generation_args)
                        # else:
                        output = pipe(messages, **generation_args)

                        if generation_args["return_full_text"]:
                            returned_text =  (output[0]["generated_text"][-1]['content'])
                        else:
                            returned_text =  output[0]['generated_text']

                        text_output = returned_text.strip()
                        # For Phi models
                        text_output = text_output.replace("```json", "").replace("```", "")
                        print("---------------------text_output-----------------")
                        print(text_output)
                        print("-------------------------------------------------")
                        dict_output = text_output.split('}')[0] + '}'

                        try:
                            tmp_dict = literal_eval(dict_output)
                        except:
                            pass

                        try:
                            save_data_single_prompt_multiple_questions(model_id, returned_text, fname, lang, lang2, run_id)
                        except:
                            tmp_dict = None

                        if tmp_dict is not None:
                            break

def main():

    # For a question list regarding one entity, we gather 3 outputs from each llm
    num_run = 3
    for model_id in [
        # "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        # "mistralai/Mistral-7B-Instruct-v0.2",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "meta-llama/Meta-Llama-3.1-70B-Instruct",
        # "google/gemma-1.1-7b-it",
        # "google/gemma-2-9b-it"
    ]:
        llm_inference(model_id, num_run)

if __name__ == "__main__":
    main()

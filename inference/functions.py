import os
import glob
import json
from ast import literal_eval

en_instruction = "##Instruction\n" \
              "Generate only one word answers.\n" \
              "Do not include any explanations.\n" \
              "Return answers in this format: { 'question number':['answer1','answer2',etc] }\n" \
              "If you do not know the answer to a question, return N/A\n"

base_path = '/apollo/dya/ISWS/data/'

# preprocess the jsons file
def create_single_prompt_multiple_questions(lang, lang2):

    src_dir = base_path + lang + '/' +  lang2 + '/raw_data/'
    tgt_dir = base_path + lang + '/' +  lang2 + '/merged/'

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
        print(f'Folder "{tgt_dir}" created.')

    files = glob.glob(os.path.join(src_dir, '*'))
    for file_path in files:

        # read the json file
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fp:
                raw_data = json.load(fp)[0]
        else:
            raise("Could not read the file.")

        # prompt construction with multiple questions
        prompt = ''
        idx_seq = []
        i = 1
        for k, v in raw_data.items():
            if v['idx'] == 0:
                continue
            if len(v['value']) == 0:
                continue
            else:
                idx_seq.append(v['idx'])
                prompt += str(i)
                prompt += '. '
                prompt += v['prompt']
                prompt += '\n'
                i += 1

        # preparations for both llm inputs (en_instruction, prompt) and evaluation( idx_seq )
        data = dict()
        data['raw_data'] = raw_data
        data['en_instruction'] = en_instruction
        data['prompt'] = prompt
        data['idx_seq'] = idx_seq

        # saving the pre-processed data
        f_name = os.path.basename(file_path)
        file_path2 = os.path.join(tgt_dir, f_name)
        with open(file_path2, 'w') as fp:
            json.dump(data, fp)

# dataloder function -> fetching data for inference
def fetch_data_single_prompt_multiple_questions(lang, lang2):

    folder_path = base_path + lang + '/' + lang2 + '/merged/'
    files = glob.glob(os.path.join(folder_path, '*'))

    for file_path in files:
        fname = os.path.basename(file_path)
        with open(file_path,'r') as fp:
            data = json.load(fp)

        en_instruction = data['en_instruction']
        prompt = data['prompt']

        yield en_instruction, prompt, fname

# saving inference outputs into json files -> based on run_id & llm choice
def save_data_single_prompt_multiple_questions(model_id, text_output, fname, lang, lang2, run_id):

    if model_id == "mistralai/Mistral-7B-Instruct-v0.2":
        model_tag = "mistral"
    elif model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
        model_tag = "llama3"
    elif model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model_tag = "llama3_1"
    elif model_id == "google/gemma-1.1-7b-it":
        model_tag = "gemma"
    elif model_id == "google/gemma-2-9b-it":
        model_tag = "gemma2"
    elif model_id == "microsoft/Phi-3-mini-4k-instruct":
        model_tag = "phi3"
    else:
        raise("Could not find the model id.")

    # create directories
    src_fpath = base_path + lang + '/'+ lang2 + '/merged/' + fname
    tgt_dir_model = base_path + lang + '/' + lang2 + '/' + model_tag
    tgt_dir = base_path + lang + '/' + lang2  + '/' + model_tag +  '/run_' + str(run_id)

    if not os.path.exists(tgt_dir_model):
        os.makedirs(tgt_dir_model)
        print(f'Folder "{tgt_dir_model}" created.')

    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
        print(f'Folder "{tgt_dir}" created.')

    with open(src_fpath, 'r') as fp:
        data = json.load(fp)

    # process outputs with idx
    text_output = text_output.strip()
    dict_output = text_output.split('}')[0] + '}'
    tmp_dict = literal_eval(dict_output)
    outputs = dict()
    for idx, answers in sorted([(int(k), v) for k, v in tmp_dict.items()], key=lambda x: x[0]):
        pos = data['idx_seq'][idx - 1]
        outputs[pos] = answers

    data['outputs'] = outputs
    data['model_id'] = model_id

    # saving outputs along with previous data into single json file
    tgt_fpath = os.path.join(tgt_dir, fname)
    with open(tgt_fpath, 'w') as fp:
        json.dump(data, fp)

# some llms do not support system role
def get_messages(model_id, en_instruction, prompt):

    if model_id in ["microsoft/Phi-3-mini-4k-instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        messages = [
            {"role": "system", "content": en_instruction },
            {"role": "user", "content": prompt },
        ]

    elif model_id in ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "google/gemma-2-9b-it"]:
        messages = [
            {"role": "user", "content": en_instruction + prompt},
        ]
    else:
        raise ("Could not find the model_id.")

    return messages

# we pre-process all the language pairs
def main():
    # entity(celebrity)'s native language
    for lang in ['en', 'de', 'it', 'pl', 'ru']:
        # prompt language
        for lang2 in ['en', 'de', 'it', 'pl', 'ru']:
            create_single_prompt_multiple_questions(lang, lang2)

if __name__ == '__main__':
    main()
import os
import glob
import json
import pprint

import torch
from torch.utils.data import Dataset

languages_dict = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
    "de": "German",
    "zh": "Mandarin",
    "ar": "Arabic",
    "pl": "Polish",
    "kn": "Kannada",
    "hi": "Hindi",
    "nl": "Dutch"
}

def collate_fn(samples):

    Q_number = [sample["Q_number"] for sample in samples]
    LoE = [sample["LoE"] for sample in samples]
    property = [sample["property"] for sample in samples]
    LoQ= [sample["LoQ"] for sample in samples]
    input = [sample["input"] for sample in samples]
    prompt = [sample["prompt"] for sample in samples]
    gt = [sample["gt"] for sample in samples]

    batch = {
        "Q_number": Q_number,
        "LoE": LoE,
        "property": property,
        "LoQ": LoQ,
        "input": input,
        "prompt": prompt,
        "gt": gt
    }
    return batch

class MultilingualQADataset(Dataset):
    def __init__(self, data_dir, lang_set, properties):
        """
        Initializes the dataset by setting up paths and transformations.
        """
        self.base_path = data_dir
        self.lang_set_LoE = lang_set
        self.lang_set_LoQ = lang_set
        self.properties = properties
        self.data = self._load_data()

    # Some llms have fine-grained roles in prompt engineering, e.g., system role, user role
    # So it is required to call this function to generate correct prompt based for certain llms
    def _set_model_id(self, model_id):
        self.model_id = model_id

    def _load_data(self):
        """
        Loads data from the directory and returns a list or other data structure.

        Returns:
            data (list): A list containing all the data samples.
        """

        data = []
        for LoE in self.lang_set_LoE:
            file_path = self.base_path + '/' + 'prompt_{}_100.json'.format(LoE)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as fp:
                    raw_data = json.load(fp)
            else:
                raise ("Could not read the file.")

            for ent_info in raw_data:
                ent_ID = ent_info['ent_ID']
                question_prompts = sorted([key for key in ent_info.keys() if key.endswith('_prompt')])
                question_ground_truths = sorted([key for key in ent_info.keys() if key.endswith('_ground_truth')])
                for LoQ in self.lang_set_LoQ:
                    for prompt_key, gt_key in zip(question_prompts, question_ground_truths):

                        assert prompt_key.replace('_prompt', '') == gt_key.replace('_ground_truth', '')

                        property = prompt_key.replace('_prompt', '')
                        prompt = ent_info[prompt_key]
                        gt = ent_info[gt_key]

                        if LoQ not in prompt:
                            continue

                        instruction, question = prompt[LoQ].replace('  ', '').split('##')[1:]

                        if property in ['dob']:
                            data.append([ent_info['ent_ID'], LoE, property, LoQ, gt, instruction, question])
                        else:
                            data.append([ent_ID, LoE, property, LoQ, gt[LoQ], instruction, question])

        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """

        Q_number, LoE, property, LoQ, gt, instruction, question = self.data[idx]

        item = dict()
        item['Q_number'] = Q_number # q number (entity)
        item['LoE'] = LoE # language of entity
        item['property'] = property # questions id
        item['LoQ'] = LoQ # language of question
        item['gt'] = gt # ground truth

        # v['instruction'] = ( f"### Instruction\n"
        #                      f"Generate only one-word answers for my question.\n"
        #                      f"Return the answers in such a list format, e.g., ['answer'] or ['answer1', 'answer2', etc] if multiple answers existed; if the answer is time-related, it should follow ['YYYY-MM-DD']; if the answer is a person's name, return the full name. \n"
        #                      f"The answers must be in {languages_dict[LoE]}.\n"
        #                      f"Do not clarify your answers.\n"
        #                      f"Do not explain yourself.\n"
        #                        )
        #
        # if self.model_id in ["microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3-mini-4k-instruct",
        #                 "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
        #                 "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        #     messages = [
        #         {"role": "system", "content": v['instruction'] },
        #         {"role": "user", "content": '### Question\nAnswer my question:' + v['prompt'] + '\n### Answer:'},
        #     ]
        #
        # elif self.model_id in ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "google/gemma-2-9b-it"]:
        #     messages = [
        #         {"role": "user", "content": v['instruction'] + '### Question\nAnswer my question:' + v['prompt'] + '\n### Answer:'},
        #     ]
        # else:
        #     raise ("Could not find the model_id.")

        if self.model_id in ["microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3-mini-4k-instruct",
                        "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
            messages = [
                {"role": "system", "content": instruction },
                {"role": "user", "content": question },
            ]

        elif self.model_id in ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "google/gemma-2-9b-it"]:
            messages = [
                {"role": "user", "content": instruction + question },
            ]
        else:
            raise ("Could not find the model_id.")

        item['input'] = messages
        item['prompt'] = instruction + question

        return item

if __name__ == '__main__':

    data_dir = '/apollo/dya/ISWS/data_v1/Prompt_final'
    lang_set = ['ar', 'en', 'de', 'fr', 'it', 'pl', 'ru', 'zh']
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    properties = ['pob', 'dob', 'occ', 'country']
    dataset = MultilingualQADataset(data_dir, lang_set, properties)
    dataset._set_model_id(model_id)
    print(len(dataset))
    pprint.pprint (dataset.__getitem__(50))


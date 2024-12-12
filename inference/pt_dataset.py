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
    LoQ= [sample["LoQ"] for sample in samples]
    LoI = [sample["LoI"] for sample in samples]
    qid = [sample["qid"] for sample in samples]
    input = [sample["input"] for sample in samples]

    batch = {
        "Q_number": Q_number,
        "LoE": LoE,
        "LoQ": LoQ,
        "LoI": LoI,
        "qid": qid,
        "input": input
    }
    return batch


class MultilingualQADataset(Dataset):
    def __init__(self, data_dir, lang_set):
        """
        Initializes the dataset by setting up paths and transformations.
        """
        self.base_path = data_dir
        self.lang_set = lang_set
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
        for LoE in self.lang_set:
            for LoQ in self.lang_set:
                # TODO
                # LoIs = ['en', LoQ] if LoQ != 'en' else [LoQ]
                LoIs = ['en']
                for LoI in LoIs:
                    # read the json file
                    src_dir = self.base_path + LoE + '/' + LoQ + '/raw_data/'
                    files = glob.glob(os.path.join(src_dir, '*'))
                    for file_path in files:
                        if os.path.isfile(file_path):
                            with open(file_path, 'r') as fp:
                                raw_data = json.load(fp)[0]
                            Q_number = os.path.basename(file_path).split('_')[0]
                        else:
                            raise ("Could not read the file.")

                        for k, v in raw_data.items():
                            if v['idx'] == 0:
                                continue
                            if len(v['value']) == 0:
                                continue
                            else:
                                data.append([Q_number, LoE, LoQ, LoI, v])
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
        item = {}

        Q_number, LoE, LoQ, LoI, v = self.data[idx]

        item['Q_number'] = Q_number # q number (entity)
        item['LoE'] = LoE # language of entity
        item['LoQ'] = LoQ # language of question
        item['LoI'] = LoI # language of instruction
        item['qid'] = v['idx'] # questions id

        v['instruction'] = ( f"### Instruction\n"
                             f"Generate only one-word answers for my question.\n"
                             f"Return the answers in such a list format, e.g., ['answer'] or ['answer1', 'answer2', etc] if multiple answers existed; if the answer is time-related, it should follow ['YYYY-MM-DD']; if the answer is a person's name, return the full name. \n"
                             f"The answers must be in {languages_dict[LoE]}.\n"
                             f"Do not clarify your answers.\n"
                             f"Do not explain yourself.\n"
                               )

        if self.model_id in ["microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3-mini-4k-instruct",
                        "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
            messages = [
                {"role": "system", "content": v['instruction'] },
                {"role": "user", "content": '### Question\nAnswer my question:' + v['prompt'] + '\n### Answer:'},
            ]

        elif self.model_id in ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-1.1-7b-it", "google/gemma-2-9b-it"]:
            messages = [
                {"role": "user", "content": v['instruction'] + '### Question\nAnswer my question:' + v['prompt'] + '\n### Answer:'},
            ]
        else:
            raise ("Could not find the model_id.")

        item['input'] = messages

        return item

if __name__ == '__main__':
    data_dir = '/apollo/dya/ISWS/data/'
    lang_set = ['en', 'de', 'it', 'pl', 'ru']
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset = MultilingualQADataset(data_dir, lang_set)
    dataset._set_model_id(model_id)
    print(len(dataset))
    pprint.pprint (dataset.__getitem__(50))


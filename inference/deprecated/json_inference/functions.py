import os
import glob
import json
import re
from ast import literal_eval

base_path = '/apollo/dya/ISWS/data/'

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

translations = {
    "en": "Return answers in {lang}",
    "fr": "Retournez les réponses en {lang}",
    "it": "Restituisci le risposte in {lang}",
    "ru": "Верните ответы на {lang}",
    "de": "Geben Sie die Antworten auf {lang} zurück",
    "zh": "以{lang}返回答案",
    "ar": "أعد الإجابات باللغة {lang}",
    "pl": "Zwróć odpowiedzi w języku {lang}",
    "kn": "ಉತ್ತರಗಳನ್ನು {lang}ದಲ್ಲಿ ಹಿಂತಿರುಗಿಸಿ",
    "hi": "उत्तर {lang} में लौटाएं",
    "nl": "Geef antwoorden in het {lang}"
}

translations_dict = {
    "en": {
        "en": "English",
        "fr": "French",
        "it": "Italian",
        "ru": "Russian",
        "de": "German",
        "zh": "Chinese",
        "ar": "Arabic",
        "pl": "Polish",
        "kn": "Kannada",
        "hi": "Hindi",
        "nl": "Dutch"
    },
    "fr": {
        "en": "anglais",
        "fr": "français",
        "it": "italien",
        "ru": "russe",
        "de": "allemand",
        "zh": "chinois",
        "ar": "arabe",  # Corrected this to French for Arabic
        "pl": "polonais",
        "kn": "kannada",  # Use Roman alphabet for consistency
        "hi": "hindi",  # Use Roman alphabet for consistency
        "nl": "néerlandais"
    },
    "it": {
        "en": "inglese",  # English
        "fr": "francese",  # French
        "it": "italiano",  # Italian
        "ru": "russo",  # Russian
        "de": "tedesco",  # German
        "zh": "cinese",  # Chinese
        "ar": "arabo",  # Arabic
        "pl": "polacco",  # Polish
        "kn": "kannada",  # Romanized Kannada for consistency
        "hi": "hindi",  # Romanized Hindi for consistency
        "nl": "olandese"  # Dutch
    },
    "ru": {
        "en": "английский",  # English
        "fr": "французский",  # French
        "it": "итальянский",  # Italian
        "ru": "русский",  # Russian
        "de": "немецкий",  # German
        "zh": "китайский",  # Chinese
        "ar": "арабский",  # Arabic
        "pl": "польский",  # Polish
        "kn": "каннада",  # Romanized Kannada for consistency
        "hi": "хинди",  # Romanized Hindi for consistency
        "nl": "голландский"  # Dutch
    },
    "de": {
        "en": "Englisch",  # English
        "fr": "Französisch",  # French
        "it": "Italienisch",  # Italian
        "ru": "Russisch",  # Russian
        "de": "Deutsch",  # German
        "zh": "Chinesisch",  # Chinese
        "ar": "Arabisch",  # Arabic
        "pl": "Polnisch",  # Polish
        "kn": "Kannada",  # Romanized Kannada for consistency
        "hi": "Hindi",  # Romanized Hindi for consistency
        "nl": "Niederländisch"  # Dutch
    },
    "zh": {
        "en": "英文",
        "fr": "法文",
        "it": "意大利文",
        "ru": "俄文",
        "de": "德文",
        "zh": "中文",
        "ar": "阿拉伯文",
        "pl": "波兰文",
        "kn": "卡纳达文",
        "hi": "印地文",
        "nl": "荷兰文"
    },
    "ar": {
        "en": "الإنجليزية",  # English
        "fr": "الفرنسية",  # French
        "it": "الإيطالية",  # Italian
        "ru": "الروسية",  # Russian
        "de": "الألمانية",  # German
        "zh": "الصينية",  # Chinese
        "ar": "العربية",  # Arabic
        "pl": "البولندية",  # Polish
        "kn": "الكانادا",  # Kannada (corrected to Romanized form in Arabic)
        "hi": "الهندية",  # Hindi (corrected to Romanized form in Arabic)
        "nl": "الهولندية"  # Dutch
    },
    "pl": {
        "en": "angielski",  # English
        "fr": "francuski",  # French
        "it": "włoski",  # Italian
        "ru": "rosyjski",  # Russian
        "de": "niemiecki",  # German
        "zh": "chiński",  # Chinese
        "ar": "arabski",  # Arabic
        "pl": "polski",  # Polish
        "kn": "kannada",  # Romanized Kannada for consistency
        "hi": "hindi",  # Romanized Hindi for consistency
        "nl": "holenderski"  # Dutch
    },
    "kn": {
        "en": "ಇಂಗ್ಲಿಷ್",  # English
        "fr": "ಫ್ರೆಂಚ್",  # French
        "it": "ಇಟಾಲಿಯನ್",  # Italian
        "ru": "ರಷ್ಯನ್",  # Russian
        "de": "ಜರ್ಮನ್",  # German
        "zh": "ಚೀನಿ",  # Chinese
        "ar": "ಅರಬಿಕ್",  # Arabic
        "pl": "ಪೋಲಿಷ್",  # Polish
        "kn": "ಕನ್ನಡ",  # Kannada
        "hi": "ಹಿಂದಿ",  # Hindi
        "nl": "ಡಚ್"  # Dutch
    },
    "hi": {
        "en": "अंग्रेज़ी",
        "fr": "फ़्रेंच",
        "it": "इतालवी",
        "ru": "रूसी",
        "de": "जर्मन",
        "zh": "चीनी",
        "ar": "अरबी",
        "pl": "पोलिश",
        "kn": "कन्नड़",
        "hi": "हिंदी",
        "nl": "डच"
    },
    "nl": {
        "en": "Engels",
        "fr": "Frans",
        "it": "Italiaans",
        "ru": "Russisch",
        "de": "Duits",
        "zh": "Chinees",
        "ar": "Arabisch",
        "pl": "Pools",
        "kn": "Kannada",
        "hi": "Hindi",
        "nl": "Nederlands"
    }
}

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

        # Returned answer must be in this language, otherwise the evaluation would be incorrect
        q_lang = lang2
        gt_lang = raw_data['authorLabel']['xml:lang']
        answer_in_lang = translations[q_lang].replace('{lang}', translations_dict[q_lang][gt_lang])

        q_lang = languages_dict[q_lang]
        gt_lang = languages_dict[gt_lang]

        if q_lang == gt_lang:
            en_instruction = ( f"##Instruction\n"
                             f"Generate only one word answers.\n"
                             f"Do not include any explanations.\n"
                             f"Return answers in this JSON format: {{ 'question number':['answer1','answer2',etc] }}, time-related answer should follow 'YYYY-MM-DD'. \n"
                             f"Return answers must be in {gt_lang}.\n"
                             f"If you do not know the answer to a question, return 'N/A'.\n"
                             f"Do not include 'N/A' in the answer list if you know partial answers to the question.\n"
                             f"{answer_in_lang}.\n"
                               )
        else:
            en_instruction = ( f"##Instruction\n"
                             f"Generate only one word answers.\n"
                             f"Do not include any explanations.\n"
                             f"Return answers in this JSON format: {{ 'question number':['answer1','answer2',etc] }}, time-related answer should follow 'YYYY-MM-DD'. \n"
                             f"Return answers must be in {gt_lang}.\n"
                             f"Do not return answers in {q_lang}! \n"
                             f"If you do not know the answer to a question, return 'N/A'.\n"
                             f"Do not include 'N/A' in the answer list if you know partial answers to the question.\n"
                             f"{answer_in_lang}.\n"
                               )

        # "mistralai/Mistral-7B-Instruct-v0.2" still uses the origin one
        en_instruction_origin = "##Instruction\n" \
                         "Generate only one word answers.\n" \
                         "Do not include any explanations.\n" \
                         "Return answers in this JSON format: { 'question number':['answer1','answer2',etc] }\n" \
                         "If you do not know the answer to a question, return N/A\n"

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
    elif model_id == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        model_tag = "llama3_1_70b"
    elif model_id == "google/gemma-1.1-7b-it":
        model_tag = "gemma"
    elif model_id == "google/gemma-2-9b-it":
        model_tag = "gemma2"
    elif model_id == "microsoft/Phi-3-mini-4k-instruct":
        model_tag = "phi3"
    elif model_id == "microsoft/Phi-3.5-mini-instruct":
        model_tag = "phi3_5"
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
    # Create a new dictionary with keys that retain the full number (single or multi-digit)
    tmp_dict = { re.findall(r'^\d+', k)[0]: v for k, v in tmp_dict.items() }
    tmp_list = sorted([(int(k), v) for k, v in tmp_dict.items()], key=lambda x: x[0])
    outputs = dict()
    for idx, answers in tmp_list:
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

    if model_id in ["microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"]:
        messages = [
            {"role": "system", "content": en_instruction + "Do not provide any note and explanation. Please only return answers in JSON. \n" },
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
    # entity language
    for lang in ['en', 'de', 'it', 'pl', 'ru']:
        # prompt language
        for lang2 in ['en', 'de', 'it', 'pl', 'ru']:
            create_single_prompt_multiple_questions(lang, lang2)

if __name__ == '__main__':
    main()
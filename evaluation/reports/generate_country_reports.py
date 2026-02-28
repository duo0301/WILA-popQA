import csv, os
from collections import defaultdict

base = 'C:/Users/andsc/Desktop/Evals/dataset_2026_sampled'
models = {
    'Gemma-3-12B': ('sample-gemma-12b', 'property_country_gemma-3-12b-it.csv'),
    'Gemma-2-9B': ('sample-gemma-9b', 'property_country_gemma-2-9b-it.csv'),
    'GLM-4-9B': ('sample-glm-9b', 'property_country_glm-4-9b-chat-hf.csv'),
    'Llama-3.1-8B': ('sample-llama-8b', 'property_country_Meta-Llama-3.1-8B-Instruct.csv'),
    'Mistral-7B': ('sample-mistral-7b', 'property_country_Mistral-7B-Instruct-v0.3.csv'),
    'Moonlight-16B': ('sample-moonlight-16b', 'property_country_Moonlight-16B-A3B-Instruct.csv'),
    'OLMo-3-7B': ('sample-olmo-7b', 'property_country_Olmo-3-7B-Instruct.csv'),
    'Phi-4': ('sample-phi-4', 'property_country_phi-4.csv'),
    'Qwen3-14B': ('sample-qwen-14b', 'property_country_Qwen3-14B.csv'),
    'Qwen3-8B': ('sample-qwen-8b', 'property_country_Qwen3-8B.csv'),
    'Nemotron-9B': ('sample-nemotron-9b', 'property_country_NVIDIA-Nemotron-Nano-9B-v2.csv'),
    'DeepSeek-V2-Lite': ('sample-deekseek-v2-lite', 'property_country_DeepSeek-V2-Lite-Chat.csv'),
}

def calc_f1(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

def load(folder, fname):
    path = os.path.join(base, folder, 'eval', fname)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# Collect data per model
all_data = {}
all_loe = set()
all_loq = set()

for model_name, (folder, fname) in models.items():
    rows = load(folder, fname)
    all_data[model_name] = rows
    for r in rows:
        all_loe.add(r['LoE'])
        all_loq.add(r['LoQ'])

all_loe = sorted(all_loe)
all_loq = sorted(all_loq)

# =====================
# REPORT 1: Overall F1 per model
# =====================
print('=' * 70)
print('REPORT 1: Overall F1 Score per Model (property_country)')
print('=' * 70)
print(f'{"Model":<20} {"TP":>6} {"FP":>6} {"FN":>6} {"Prec":>10} {"Recall":>10} {"F1":>10}')
print('-' * 70)
for model_name in models:
    tp = sum(1 for r in all_data[model_name] if r['eval'] == 'TP')
    fp = sum(1 for r in all_data[model_name] if r['eval'] == 'FP')
    fn = sum(1 for r in all_data[model_name] if r['eval'] == 'FN')
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f = calc_f1(tp, fp, fn)
    print(f'{model_name:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>10.4f} {rec:>10.4f} {f:>10.4f}')

# =====================
# REPORT 2: F1 by LoE (Language of Entity) per model
# =====================
print()
print('=' * 80)
print('REPORT 2: F1 Score by Language of Entity (LoE) per Model')
print('=' * 80)
header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loe)
print(header)
print('-' * len(header))
for model_name in models:
    scores = {}
    for loe in all_loe:
        subset = [r for r in all_data[model_name] if r['LoE'] == loe]
        tp = sum(1 for r in subset if r['eval'] == 'TP')
        fp = sum(1 for r in subset if r['eval'] == 'FP')
        fn = sum(1 for r in subset if r['eval'] == 'FN')
        scores[loe] = calc_f1(tp, fp, fn)
    print(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loe))

# =====================
# REPORT 3: F1 by LoQ (Language of Query) per model
# =====================
print()
print('=' * 80)
print('REPORT 3: F1 Score by Language of Query (LoQ) per Model')
print('=' * 80)
header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loq)
print(header)
print('-' * len(header))
for model_name in models:
    scores = {}
    for loq in all_loq:
        subset = [r for r in all_data[model_name] if r['LoQ'] == loq]
        tp = sum(1 for r in subset if r['eval'] == 'TP')
        fp = sum(1 for r in subset if r['eval'] == 'FP')
        fn = sum(1 for r in subset if r['eval'] == 'FN')
        scores[loq] = calc_f1(tp, fp, fn)
    print(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loq))

# =====================
# REPORT 4: F1 by LoE x LoQ matrix (averaged across all models)
# =====================
print()
print('=' * 80)
print('REPORT 4: F1 by LoE x LoQ (averaged across all models)')
print('=' * 80)
lbl = 'LoE\\LoQ'
header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
print(header)
print('-' * len(header))
for loe in all_loe:
    row_vals = []
    for loq in all_loq:
        model_f1s = []
        for model_name in models:
            subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
            tp = sum(1 for r in subset if r['eval'] == 'TP')
            fp = sum(1 for r in subset if r['eval'] == 'FP')
            fn = sum(1 for r in subset if r['eval'] == 'FN')
            model_f1s.append(calc_f1(tp, fp, fn))
        row_vals.append(sum(model_f1s) / len(model_f1s))
    print(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))

# =====================
# REPORT 5: Per-model LoE x LoQ F1 matrices
# =====================
for model_name in models:
    print()
    print('=' * 80)
    print(f'REPORT 5: F1 Matrix LoE x LoQ - {model_name}')
    print('=' * 80)
    header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
    print(header)
    print('-' * len(header))
    for loe in all_loe:
        row_vals = []
        for loq in all_loq:
            subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
            tp = sum(1 for r in subset if r['eval'] == 'TP')
            fp = sum(1 for r in subset if r['eval'] == 'FP')
            fn = sum(1 for r in subset if r['eval'] == 'FN')
            row_vals.append(calc_f1(tp, fp, fn))
        print(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))

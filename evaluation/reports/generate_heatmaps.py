import csv, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_root = os.path.join(SCRIPT_DIR, 'heatmaps')

PROP_FOLDERS = {
    'pob':     'place_of_birth_evals',
    'dob':     'date_of_birth_evals',
    'country': 'country_evals',
}

MODEL_SUFFIXES = {
    'Gemma-3-12B':    'gemma-3-12b-it',
    'Gemma-2-9B':     'gemma-2-9b-it',
    'GLM-4-9B':       'glm-4-9b-chat-hf',
    'Llama-3.1-8B':   'Meta-Llama-3.1-8B-Instruct',
    'Mistral-7B':     'Mistral-7B-Instruct-v0.3',
    'Moonlight-16B':  'Moonlight-16B-A3B-Instruct',
    'OLMo-3-7B':      'Olmo-3-7B-Instruct',
    'Phi-4':          'phi-4',
    'Qwen3-14B':      'Qwen3-14B',
    'Qwen3-8B':       'Qwen3-8B',
    'Nemotron-9B':    'NVIDIA-Nemotron-Nano-9B-v2',
    'DeepSeek-V2-Lite': 'DeepSeek-V2-Lite-Chat',
}

PROP_LABELS = {
    'pob':     'Place of Birth',
    'country': 'Country of Citizenship',
    'dob':     'Date of Birth',
}

WEIGHTS = {
    'pob': {
        'exact_match':      1.0,
        'country_match':    0.8,
        'historical_match': 0.7,
    },
    'country': {
        'exact_match':      1.0,
        'alias_match':      1.0,
        'demonym_match':    1.0,
        'historical_match': 0.7,
        'substring_match':  0.9,
    },
    'dob': {
        'exact_match':           1.0,
        'swap_match':            1.0,
        'year_month_match':      0.7,
        'year_month_swap_match': 0.7,
    },
}

# Properties that have _prov CSV files (with auto/judge provenance columns)
PROPS_WITH_PROV = {'pob', 'country'}


# ── helpers ──────────────────────────────────────────────────────────────────

def calc_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def load(suffix, prop, prov=False):
    fname = f'property_{prop}_{suffix}{"_prov" if prov else ""}.csv'
    path  = os.path.join(SCRIPT_DIR, PROP_FOLDERS[prop], fname)
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def f1_of(rows, prov_filter=None):
    if prov_filter:
        rows = [r for r in rows if r.get('provenance') == prov_filter]
    tp = sum(1 for r in rows if r['eval'] == 'TP')
    fp = sum(1 for r in rows if r['eval'] == 'FP')
    fn = sum(1 for r in rows if r['eval'] == 'FN')
    return calc_f1(tp, fp, fn)


def weighted_f1_of(rows, prop, prov_filter=None):
    if prov_filter:
        rows = [r for r in rows if r.get('provenance') == prov_filter]
    w = WEIGHTS.get(prop, {})
    w_tp = sum(w.get(r['eval_type'], 0.0) for r in rows if r['eval'] == 'TP')
    fp   = sum(1 for r in rows if r['eval'] == 'FP')
    fn   = sum(1 for r in rows if r['eval'] == 'FN')
    prec = w_tp / (w_tp + fp) if (w_tp + fp) > 0 else 0
    rec  = w_tp / (w_tp + fn) if (w_tp + fn) > 0 else 0
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0


def save_heatmap(matrix, row_labels, col_labels, title, path,
                 vmin=0.0, vmax=1.0, figsize=None):
    rows, cols = len(row_labels), len(col_labels)
    if figsize is None:
        figsize = (max(6, cols * 0.9 + 1.5), max(4, rows * 0.65 + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True, fmt='.2f',
        xticklabels=col_labels, yticklabels=row_labels,
        cmap='RdYlGn', vmin=vmin, vmax=vmax,
        linewidths=0.4, linecolor='#ddd',
        ax=ax,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  saved {os.path.relpath(path, out_root)}')


# ── per-property generation ───────────────────────────────────────────────────

def generate(prop):
    prop_label = PROP_LABELS.get(prop, prop)
    print(f'\n=== {prop_label} ===')

    has_prov = prop in PROPS_WITH_PROV

    # load files for all models (prov files if available, regular otherwise)
    all_data = {}
    all_loe, all_loq = set(), set()

    for model, suffix in MODEL_SUFFIXES.items():
        rows = load(suffix, prop, prov=has_prov)
        all_data[model] = rows
        for r in rows:
            all_loe.add(r['LoE'])
            all_loq.add(r['LoQ'])

    loe_list   = sorted(all_loe)
    loq_list   = sorted(all_loq)
    model_list = list(MODEL_SUFFIXES.keys())

    prop_dir      = os.path.join(out_root, prop)
    per_model_dir = os.path.join(prop_dir, 'per_model')

    # helpers: build matrices (unweighted and weighted)
    def loe_loq_matrix(rows, prov_filter=None, weighted=False):
        mat = np.zeros((len(loe_list), len(loq_list)))
        for i, loe in enumerate(loe_list):
            for j, loq in enumerate(loq_list):
                subset = [r for r in rows if r['LoE'] == loe and r['LoQ'] == loq]
                mat[i, j] = (weighted_f1_of(subset, prop, prov_filter)
                             if weighted else f1_of(subset, prov_filter))
        return mat

    def model_lang_matrix(lang_key, lang_list, prov_filter=None, weighted=False):
        mat = np.zeros((len(model_list), len(lang_list)))
        for i, model in enumerate(model_list):
            for j, lang in enumerate(lang_list):
                subset = [r for r in all_data[model] if r[lang_key] == lang]
                mat[i, j] = (weighted_f1_of(subset, prop, prov_filter)
                             if weighted else f1_of(subset, prov_filter))
        return mat

    prov_variants = [(None, ''), ('auto', '_auto'), ('judge', '_judge')] if has_prov else [(None, '')]

    # ── 1. LoE×LoQ averaged across models ──────────────────────────────────
    for prov_filter, tag in prov_variants:
        pv = f' ({prov_filter})' if prov_filter else ''
        for weighted, wtag in [(False, ''), (True, '_weighted')]:
            mats = [loe_loq_matrix(all_data[m], prov_filter, weighted) for m in model_list]
            avg  = np.mean(mats, axis=0)
            wlbl = ' [weighted]' if weighted else ''
            save_heatmap(
                avg, loe_list, loq_list,
                title=f'{prop_label} — F1{wlbl}: LoE × LoQ  (avg across models{pv})',
                path=os.path.join(prop_dir, f'loe_loq_avg{tag}{wtag}.png'),
            )

    # ── 2. LoE×LoQ per model ───────────────────────────────────────────────
    for model in model_list:
        for prov_filter, tag in prov_variants:
            pv = f' ({prov_filter})' if prov_filter else ''
            for weighted, wtag in [(False, ''), (True, '_weighted')]:
                mat  = loe_loq_matrix(all_data[model], prov_filter, weighted)
                wlbl = ' [weighted]' if weighted else ''
                save_heatmap(
                    mat, loe_list, loq_list,
                    title=f'{prop_label} — F1{wlbl}: LoE × LoQ  {model}{pv}',
                    path=os.path.join(per_model_dir, f'{model}_loe_loq{tag}{wtag}.png'),
                )

    # ── 3. Model × LoE ─────────────────────────────────────────────────────
    for prov_filter, tag in prov_variants:
        pv = f' ({prov_filter})' if prov_filter else ''
        for weighted, wtag in [(False, ''), (True, '_weighted')]:
            mat  = model_lang_matrix('LoE', loe_list, prov_filter, weighted)
            wlbl = ' [weighted]' if weighted else ''
            save_heatmap(
                mat, model_list, loe_list,
                title=f'{prop_label} — F1{wlbl}: Model × LoE{pv}',
                path=os.path.join(prop_dir, f'model_loe{tag}{wtag}.png'),
                figsize=(10, 6),
            )

    # ── 4. Model × LoQ ─────────────────────────────────────────────────────
    for prov_filter, tag in prov_variants:
        pv = f' ({prov_filter})' if prov_filter else ''
        for weighted, wtag in [(False, ''), (True, '_weighted')]:
            mat  = model_lang_matrix('LoQ', loq_list, prov_filter, weighted)
            wlbl = ' [weighted]' if weighted else ''
            save_heatmap(
                mat, model_list, loq_list,
                title=f'{prop_label} — F1{wlbl}: Model × LoQ{pv}',
                path=os.path.join(prop_dir, f'model_loq{tag}{wtag}.png'),
                figsize=(10, 6),
            )


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    props = sys.argv[1:] or ['pob', 'country']
    for prop in props:
        generate(prop)
    print('\nDone.')

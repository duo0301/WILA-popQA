import csv, os, sys

base = 'C:/Users/andsc/Desktop/Evals/dataset_2026_sampled'

# eval_type weights per property (FP / FN / unknown → 0.0)
WEIGHTS = {
    'pob': {
        'exact_match':    1.0,
        'country_match':  0.8,
        'historical_match': 0.7,
    },
    'country': {
        'exact_match':    1.0,
        'alias_match':    1.0,
        'demonym_match':  1.0,
        'historical_match': 0.7,
        'substring_match': 0.9,
    },
    'dob': {
        'exact_match':           1.0,
        'swap_match':            1.0,
        'year_month_match':      0.7,
        'year_month_swap_match': 0.7,
    },
}

# property prefix -> (folder, filename) per model
MODEL_SUFFIXES = {
    'Gemma-3-12B': ('sample-gemma-12b', 'gemma-3-12b-it'),
    'Gemma-2-9B': ('sample-gemma-9b', 'gemma-2-9b-it'),
    'GLM-4-9B': ('sample-glm-9b', 'glm-4-9b-chat-hf'),
    'Llama-3.1-8B': ('sample-llama-8b', 'Meta-Llama-3.1-8B-Instruct'),
    'Mistral-7B': ('sample-mistral-7b', 'Mistral-7B-Instruct-v0.3'),
    'Moonlight-16B': ('sample-moonlight-16b', 'Moonlight-16B-A3B-Instruct'),
    'OLMo-3-7B': ('sample-olmo-7b', 'Olmo-3-7B-Instruct'),
    'Phi-4': ('sample-phi-4', 'phi-4'),
    'Qwen3-14B': ('sample-qwen-14b', 'Qwen3-14B'),
    'Qwen3-8B': ('sample-qwen-8b', 'Qwen3-8B'),
    'Nemotron-9B': ('sample-nemotron-9b', 'NVIDIA-Nemotron-Nano-9B-v2'),
    'DeepSeek-V2-Lite': ('sample-deekseek-v2-lite', 'DeepSeek-V2-Lite-Chat'),
}

def calc_f1(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    return prec, rec, 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

def calc_weighted_f1(rows, prop):
    w = WEIGHTS.get(prop, {})
    w_tp = sum(w.get(r['eval_type'], 0.0) for r in rows if r['eval'] == 'TP')
    fp   = sum(1 for r in rows if r['eval'] == 'FP')
    fn   = sum(1 for r in rows if r['eval'] == 'FN')
    prec = w_tp / (w_tp + fp) if (w_tp + fp) > 0 else 0
    rec  = w_tp / (w_tp + fn) if (w_tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return prec, rec, f1

def load(folder, fname, prov=False):
    if prov:
        fname = fname.rsplit('.', 1)[0] + '_prov.csv'
    path = os.path.join(base, folder, 'eval', fname)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def generate_reports(prop, prov=False, out=None):
    """Generate all F1 reports for a given property (e.g. 'pob', 'dob', 'country').
    If prov=True, loads _prov files and adds a provenance breakdown report.
    If out is a file object, writes there instead of stdout.
    """
    import builtins
    _print = builtins.print
    def p(*args, **kwargs):
        if out:
            kwargs.setdefault('file', out)
        _print(*args, **kwargs)

    prop_label = {
        'pob': 'Place of Birth',
        'dob': 'Date of Birth',
        'country': 'Country of Citizenship',
        'occ': 'Occupation',
    }.get(prop, prop)

    # Build model dict for this property
    models = {}
    for model_name, (folder, suffix) in MODEL_SUFFIXES.items():
        fname = f'property_{prop}_{suffix}.csv'
        models[model_name] = (folder, fname)

    # Collect data
    all_data = {}
    all_loe = set()
    all_loq = set()
    all_prov = set()

    for model_name, (folder, fname) in models.items():
        rows = load(folder, fname, prov=prov)
        all_data[model_name] = rows
        for r in rows:
            all_loe.add(r['LoE'])
            all_loq.add(r['LoQ'])
            if prov and 'provenance' in r:
                all_prov.add(r['provenance'])

    all_loe = sorted(all_loe)
    all_loq = sorted(all_loq)
    all_prov = sorted(p_ for p_ in all_prov if p_)  # exclude empty string

    prov_tag = ' [_prov]' if prov else ''

    # REPORT 1: Overall F1 per model
    p('=' * 70)
    p(f'REPORT 1: Overall F1 Score per Model (property_{prop}{prov_tag} - {prop_label})')
    p('=' * 70)
    p(f'{"Model":<20} {"TP":>6} {"FP":>6} {"FN":>6} {"Prec":>10} {"Recall":>10} {"F1":>10}')
    p('-' * 70)
    for model_name in models:
        tp = sum(1 for r in all_data[model_name] if r['eval'] == 'TP')
        fp = sum(1 for r in all_data[model_name] if r['eval'] == 'FP')
        fn = sum(1 for r in all_data[model_name] if r['eval'] == 'FN')
        prec, rec, f = calc_f1(tp, fp, fn)
        p(f'{model_name:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>10.4f} {rec:>10.4f} {f:>10.4f}')

    # REPORT 2: F1 by LoE per model
    p()
    p('=' * 80)
    p(f'REPORT 2: F1 by Language of Entity (LoE) per Model [{prop_label}{prov_tag}]')
    p('=' * 80)
    header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loe)
    p(header)
    p('-' * len(header))
    for model_name in models:
        scores = {}
        for loe in all_loe:
            subset = [r for r in all_data[model_name] if r['LoE'] == loe]
            tp = sum(1 for r in subset if r['eval'] == 'TP')
            fp = sum(1 for r in subset if r['eval'] == 'FP')
            fn = sum(1 for r in subset if r['eval'] == 'FN')
            _, _, scores[loe] = calc_f1(tp, fp, fn)
        p(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loe))

    # REPORT 3: F1 by LoQ per model
    p()
    p('=' * 80)
    p(f'REPORT 3: F1 by Language of Query (LoQ) per Model [{prop_label}{prov_tag}]')
    p('=' * 80)
    header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loq)
    p(header)
    p('-' * len(header))
    for model_name in models:
        scores = {}
        for loq in all_loq:
            subset = [r for r in all_data[model_name] if r['LoQ'] == loq]
            tp = sum(1 for r in subset if r['eval'] == 'TP')
            fp = sum(1 for r in subset if r['eval'] == 'FP')
            fn = sum(1 for r in subset if r['eval'] == 'FN')
            _, _, scores[loq] = calc_f1(tp, fp, fn)
        p(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loq))

    # REPORT 4: F1 by LoE x LoQ (averaged across all models)
    p()
    p('=' * 80)
    p(f'REPORT 4: F1 by LoE x LoQ averaged across all models [{prop_label}{prov_tag}]')
    p('=' * 80)
    lbl = 'LoE\\LoQ'
    header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
    p(header)
    p('-' * len(header))
    for loe in all_loe:
        row_vals = []
        for loq in all_loq:
            model_f1s = []
            for model_name in models:
                subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
                tp = sum(1 for r in subset if r['eval'] == 'TP')
                fp = sum(1 for r in subset if r['eval'] == 'FP')
                fn = sum(1 for r in subset if r['eval'] == 'FN')
                _, _, f = calc_f1(tp, fp, fn)
                model_f1s.append(f)
            row_vals.append(sum(model_f1s) / len(model_f1s))
        p(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))

    # REPORT 5: Per-model LoE x LoQ matrices
    for model_name in models:
        p()
        p('=' * 80)
        p(f'REPORT 5: F1 Matrix LoE x LoQ - {model_name} [{prop_label}{prov_tag}]')
        p('=' * 80)
        header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
        p(header)
        p('-' * len(header))
        for loe in all_loe:
            row_vals = []
            for loq in all_loq:
                subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
                tp = sum(1 for r in subset if r['eval'] == 'TP')
                fp = sum(1 for r in subset if r['eval'] == 'FP')
                fn = sum(1 for r in subset if r['eval'] == 'FN')
                _, _, f = calc_f1(tp, fp, fn)
                row_vals.append(f)
            p(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))

    # REPORT 6: F1 by Provenance per model (only when loading _prov files)
    if prov and all_prov:
        p()
        p('=' * 80)
        p(f'REPORT 6: F1/Prec/Recall by Provenance per Model [{prop_label}{prov_tag}]')
        p('=' * 80)
        col_w = 28
        header = f'{"Model":<20}' + ''.join(f'{pv:>{col_w}}' for pv in all_prov)
        p(header)
        p('-' * len(header))
        p(f'{"":20}' + ''.join(f'{"TP/FP/FN  Prec  Rec  F1":>{col_w}}' for _ in all_prov))
        p('-' * len(header))
        for model_name in models:
            cells = []
            for pv in all_prov:
                subset = [r for r in all_data[model_name] if r.get('provenance') == pv]
                tp = sum(1 for r in subset if r['eval'] == 'TP')
                fp = sum(1 for r in subset if r['eval'] == 'FP')
                fn = sum(1 for r in subset if r['eval'] == 'FN')
                prec, rec, f = calc_f1(tp, fp, fn)
                cells.append(f'{tp}/{fp}/{fn}  {prec:.3f}  {rec:.3f}  {f:.3f}')
            p(f'{model_name:<20}' + ''.join(f'{c:>{col_w}}' for c in cells))

        # REPORT 6b: Per-provenance overall summary table
        p()
        p('=' * 70)
        p(f'REPORT 6b: Overall P/R/F1 per Model per Provenance [{prop_label}{prov_tag}]')
        p('=' * 70)
        for pv in all_prov:
            p()
            p(f'  Provenance = "{pv}"')
            p(f'  {"Model":<20} {"TP":>6} {"FP":>6} {"FN":>6} {"Prec":>10} {"Recall":>10} {"F1":>10}')
            p('  ' + '-' * 68)
            for model_name in models:
                subset = [r for r in all_data[model_name] if r.get('provenance') == pv]
                tp = sum(1 for r in subset if r['eval'] == 'TP')
                fp = sum(1 for r in subset if r['eval'] == 'FP')
                fn = sum(1 for r in subset if r['eval'] == 'FN')
                prec, rec, f = calc_f1(tp, fp, fn)
                p(f'  {model_name:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>10.4f} {rec:>10.4f} {f:>10.4f}')

    # Weighted reports (always, since eval_type is present in both prov and non-prov files)
    _weighted_reports(prop, all_data, models, all_loe, all_loq, prov_tag, p)


def _weighted_reports(prop, all_data, models, all_loe, all_loq, prov_tag, p):
    """Weighted P/R/F1 reports (7-11), mirroring reports 1-5."""

    lbl = 'LoE\\LoQ'

    # REPORT 7: Weighted overall P/R/F1 per model
    p()
    p('=' * 70)
    p(f'REPORT 7: Weighted Overall P/R/F1 per Model (property_{prop}{prov_tag})')
    p('=' * 70)
    p(f'{"Model":<20} {"wPrec":>10} {"wRecall":>10} {"wF1":>10}')
    p('-' * 52)
    for model_name in models:
        prec, rec, f = calc_weighted_f1(all_data[model_name], prop)
        p(f'{model_name:<20} {prec:>10.4f} {rec:>10.4f} {f:>10.4f}')

    # REPORT 8: Weighted F1 by LoE per model
    p()
    p('=' * 80)
    p(f'REPORT 8: Weighted F1 by LoE per Model [{prop}{prov_tag}]')
    p('=' * 80)
    header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loe)
    p(header)
    p('-' * len(header))
    for model_name in models:
        scores = {}
        for loe in all_loe:
            subset = [r for r in all_data[model_name] if r['LoE'] == loe]
            _, _, scores[loe] = calc_weighted_f1(subset, prop)
        p(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loe))

    # REPORT 9: Weighted F1 by LoQ per model
    p()
    p('=' * 80)
    p(f'REPORT 9: Weighted F1 by LoQ per Model [{prop}{prov_tag}]')
    p('=' * 80)
    header = f'{"Model":<20}' + ''.join(f'{l:>8}' for l in all_loq)
    p(header)
    p('-' * len(header))
    for model_name in models:
        scores = {}
        for loq in all_loq:
            subset = [r for r in all_data[model_name] if r['LoQ'] == loq]
            _, _, scores[loq] = calc_weighted_f1(subset, prop)
        p(f'{model_name:<20}' + ''.join(f'{scores[l]:>8.4f}' for l in all_loq))

    # REPORT 10: Weighted F1 by LoE × LoQ averaged across models
    p()
    p('=' * 80)
    p(f'REPORT 10: Weighted F1 by LoE × LoQ averaged across models [{prop}{prov_tag}]')
    p('=' * 80)
    header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
    p(header)
    p('-' * len(header))
    for loe in all_loe:
        row_vals = []
        for loq in all_loq:
            model_f1s = []
            for model_name in models:
                subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
                _, _, f = calc_weighted_f1(subset, prop)
                model_f1s.append(f)
            row_vals.append(sum(model_f1s) / len(model_f1s))
        p(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))

    # REPORT 11: Weighted F1 Matrix LoE × LoQ per model
    for model_name in models:
        p()
        p('=' * 80)
        p(f'REPORT 11: Weighted F1 Matrix LoE × LoQ - {model_name} [{prop}{prov_tag}]')
        p('=' * 80)
        header = f'{lbl:<8}' + ''.join(f'{l:>8}' for l in all_loq)
        p(header)
        p('-' * len(header))
        for loe in all_loe:
            row_vals = []
            for loq in all_loq:
                subset = [r for r in all_data[model_name] if r['LoE'] == loe and r['LoQ'] == loq]
                _, _, f = calc_weighted_f1(subset, prop)
                row_vals.append(f)
            p(f'{loe:<8}' + ''.join(f'{v:>8.4f}' for v in row_vals))


if __name__ == '__main__':
    args = sys.argv[1:]
    prop = 'country'
    prov = False
    save = False

    for a in args:
        if a == '--prov':
            prov = True
        elif a == '--save':
            save = True
        else:
            prop = a

    report_files = {
        'pob': 'pob_f1_reports.txt',
        'country': 'country_f1_reports.txt',
        'dob': 'dob_f1_reports.txt',
    }

    root = 'C:/Users/andsc/Desktop/Evals'

    if save and prop in report_files:
        fname = report_files[prop]
        if prov:
            fname = fname.replace('_reports.txt', '_prov_reports.txt')
        fpath = os.path.join(root, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            generate_reports(prop, prov=prov, out=f)
        print(f'Saved to {fpath}')
    else:
        generate_reports(prop, prov=prov)

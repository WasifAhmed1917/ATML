"""Verify fixed Task 1 hashes, IDs, counts, balance, and truncation diagnostics."""
import argparse
from collections import Counter
import json
from common.data import load_yaml, repo_path, read_jsonl
from task1_dpo.encoding import pair_diagnostics, overflow_mode
from task1_dpo.utils import file_hash, write_json, artifact_path, identity


def validate_data(config='configs/dpo.yaml', tokenize=False):
    cfg = load_yaml(config)
    sha = json.loads(repo_path('manifests/sha256.json').read_text())
    expected = {'dpo_standard_train':1500, 'dpo_standard_eval':300,
                'dpo_length_train':1500, 'dpo_length_eval':246, 'word_limit_prompts':10}
    report = {}; diagnostics = []
    tok = None
    if tokenize:
        from common.models import load_tokenizer
        tok = load_tokenizer(cfg['base_model'])
    for key, count in expected.items():
        path = cfg['paths'][key]; rows = read_jsonl(path)
        if len(rows) != count:
            raise ValueError(f'{path}: expected {count} rows, found {len(rows)}')
        if len({r['prompt_id'] for r in rows}) != len(rows):
            raise ValueError(f'{path}: duplicate prompt IDs')
        actual = file_hash(path)
        if path in sha and actual != sha[path]:
            raise ValueError(f'{path}: course SHA256 mismatch')
        if key != 'word_limit_prompts' and path not in sha:
            raise ValueError(f'{path}: missing course hash')
        if key == 'word_limit_prompts':
            from common.metrics import parse_word_limit
            if any(parse_word_limit('\n'.join(m['content'] for m in r['messages'])) is None for r in rows):
                raise ValueError('Unparseable fixed word-limit prompt')
        strata = Counter(r.get('length_stratum') for r in rows)
        if key in ('dpo_length_train', 'dpo_length_eval'):
            n = 500 if key == 'dpo_length_train' else 82
            if strata != Counter({'preferred_longer':n, 'length_matched':n, 'rejected_longer':n}):
                raise ValueError('Fixed length strata are not balanced')
        entry = {'n':len(rows), 'sha256':actual, 'course_hash_verified':path in sha,
                 'strata':{str(k):v for k,v in strata.items()}}
        if tok and key != 'word_limit_prompts':
            ds = [{**identity(r,i), **pair_diagnostics(tok,r,int(cfg['max_sequence_length']))}
                  for i,r in enumerate(rows)]
            diagnostics.extend({'dataset':key, **d} for d in ds)
            entry.update(truncation_summary(ds))
            entry['truncation_by_stratum'] = {
                s:truncation_summary([d for d in ds if d.get('length_stratum') == s])
                for s in sorted(set(d['length_stratum'] for d in ds if 'length_stratum' in d))}
            if key == 'dpo_standard_train':
                report['beta_first_600'] = truncation_summary(ds[:int(cfg['short_ablation_examples'])])
        report[key] = entry
    dest = artifact_path(cfg['results_dir']); dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'fixed_data_manifest.json', report)
    write_json(dest/'beta_subset_ids.json', [identity(r,i) for i,r in enumerate(
        read_jsonl(cfg['paths']['dpo_standard_train'])[:cfg['short_ablation_examples']])])
    if tokenize:
        write_json(dest/'preprocessing_diagnostics.json', {'preprocessing':overflow_mode(), 'pairs':diagnostics})
        import pandas as pd
        pd.DataFrame(diagnostics).to_csv(dest/'preprocessing_diagnostics.csv', index=False)
    print(json.dumps(report, indent=2))
    return report


def truncation_summary(rows):
    return {'n':len(rows), 'overlength_pairs':sum(r['pair_overlength'] for r in rows),
            'prompt_alone_overflow':sum(r['prompt_overflow'] for r in rows),
            'prompt_truncated_pairs':sum(r['prompt_tokens_removed']>0 for r in rows),
            'response_truncated_pairs':sum(r['chosen_tokens_removed']>0 or r['rejected_tokens_removed']>0 for r in rows),
            'max_prompt_tokens':max(r['prompt_tokens_original'] for r in rows),
            'max_original_sequence_tokens':max(r['prompt_tokens_original']+max(r['chosen_tokens_original'],r['rejected_tokens_original']) for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/dpo.yaml'); p.add_argument('--tokenize', action='store_true')
    a = p.parse_args(); validate_data(a.config, a.tokenize)
if __name__ == '__main__': main()

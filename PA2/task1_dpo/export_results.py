"""Rebuild CSVs and paired review records using saved results; no model loads."""
import argparse
import json
from pathlib import Path
import pandas as pd
from common.data import load_yaml, read_jsonl
from task1_dpo.utils import artifact_path,load_records,write_json


def export_results(config='configs/dpo.yaml',require_complete=False):
    cfg=load_yaml(config); root=artifact_path(cfg['results_dir'])
    summary=[];strata=[];words=[]
    for path in sorted(root.glob('*/evaluation_summary.json')):
        s=json.loads(path.read_text())
        if s.get('smoke'):continue
        summary.append({k:v for k,v in s.items() if k not in ('strata','word_limit')})
        for label,values in s.get('strata',{}).items():
            strata.append({'condition':s['condition'],'length_stratum':label,**values})
        if 'word_limit' in s:words.append({'condition':s['condition'],**s['word_limit']})
    if not summary:raise ValueError('No full-run evaluation results to export')
    pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
    if strata:pd.DataFrame(strata).to_csv(root/'length_strata.csv',index=False)
    if words:pd.DataFrame(words).to_csv(root/'word_limit_summary.csv',index=False)
    for p in root.glob('*/training_log.jsonl'):
        records=load_records(p)
        if records:pd.DataFrame(records).to_csv(p.with_suffix('.csv'),index=False)
    # These are review material, not automatically claimed correctness/quality judgments.
    conditions=('sft','standard','beta_0.03','beta_0.10','beta_0.30','length_balanced')
    joined={}
    for condition in conditions:
        for filename,set_name in [('generations.jsonl','standard_eval'),('word_limit_generations.jsonl','word_limit')]:
            for r in load_records(root/condition/filename):
                key=(set_name,r['prompt_id'])
                entry=joined.setdefault(key,{'evaluation_set':set_name,'prompt_id':r['prompt_id'],
                    'messages':r['messages'],'manual_quality_notes':'','include_in_report':False})
                entry[condition]={k:r[k] for k in ('response','reward_score','response_length',
                                   'sampled_kl_token_mean','word_limit','word_limit_compliant','truncated')}
    dest=root/'qualitative_review.jsonl'
    # Retain user-written annotations when numerical tables are regenerated.
    old={(r['evaluation_set'],r['prompt_id']):r for r in load_records(dest)}
    for key,entry in joined.items():
        if key in old:
            for field in ('manual_quality_notes','include_in_report'):entry[field]=old[key].get(field,entry[field])
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in joined.values()),encoding='utf-8')
    required={'standard','beta_0.03','beta_0.10','beta_0.30','length_balanced'}
    present={r['condition'] for r in summary}
    checks={'required_conditions_present':required<=present,
            'both_length_comparisons_present':{r['condition'] for r in words}>={'standard','length_balanced'},
            'all_six_stratum_rows_present':len([r for r in strata if r['condition'] in ('standard','length_balanced')])==6}
    fixed = read_jsonl(cfg['paths']['dpo_standard_eval'])
    fixed_length = read_jsonl(cfg['paths']['dpo_length_eval'])
    fixed_words = read_jsonl(cfg['paths']['word_limit_prompts'])
    def matches(records, expected):
        return (len(records)==len(expected) and all(
            r['row_index']==i and r['prompt_id']==expected[i]['prompt_id']
            for i,r in enumerate(records)))
    checks['fixed_pair_and_generation_ids'] = all(
        matches(load_records(root/c/'standard_pairs.jsonl'),fixed) and
        matches(load_records(root/c/'generations.jsonl'),fixed) for c in required)
    checks['fixed_length_and_word_limit_ids'] = all(
        matches(load_records(root/c/'length_pairs.jsonl'),fixed_length) and
        matches(load_records(root/c/'word_limit_generations.jsonl'),fixed_words)
        for c in ('standard','length_balanced'))
    budgets = {'standard':1500,'length_balanced':1500,
               'beta_0.03':600,'beta_0.10':600,'beta_0.30':600}
    checks['training_budgets_and_completed_evaluations'] = all(
        json.loads((root/c/'run_metadata.json').read_text()).get('train_examples') == n and
        json.loads((root/c/'run_metadata.json').read_text()).get('status') == 'complete' and
        json.loads((root/c/'evaluation_metadata.json').read_text()).get('status') == 'complete'
        if (root/c/'run_metadata.json').exists() and (root/c/'evaluation_metadata.json').exists() else False
        for c,n in budgets.items())
    write_json(root/'completion_check.json',checks)
    if require_complete and not all(checks.values()):raise ValueError(f'Incomplete Task 1 results: {checks}')
    print(f'Exported CSV tables and qualitative review records to {root}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/dpo.yaml')
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();export_results(a.config,a.require_complete)
if __name__=='__main__':main()

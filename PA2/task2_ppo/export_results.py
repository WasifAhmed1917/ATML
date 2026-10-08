"""Validate every required PPO condition and export numeric evidence and manual review inputs."""
import argparse
import json
import pandas as pd
from common.data import load_yaml,read_jsonl
from task2_ppo.runtime import artifact_path,write_json,load_records


def export(config='configs/ppo.yaml'):
    cfg=load_yaml(config);root=artifact_path(cfg['results_dir']);train=read_jsonl(cfg['paths']['rl_prompt_train']);evalrows=read_jsonl(cfg['paths']['rl_prompt_eval'])
    expected={'standard':(20,.2,.1),**{f'clip_{e:.2f}':(8,e,.1) for e in cfg['clip_values']},**{f'kl_{b:.2f}':(8,.2,b) for b in cfg['kl_values']}}
    summaries=[];histories=[];review=[]
    for name in ['midpoint',*expected]:
        folder=root/name;em=json.loads((folder/'evaluation_metadata.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        records=load_records(folder/'generations.jsonl')
        if em['status']!='complete' or em['smoke'] or len(records)!=len(evalrows):raise ValueError(f'Incomplete evaluation: {name}')
        if [r['prompt_id'] for r in records]!=[r['prompt_id'] for r in evalrows]:raise ValueError(f'Evaluation prompt IDs differ: {name}')
        if name in expected:
            updates,eps,beta=expected[name];out=artifact_path(f'outputs/task2_ppo/{name}')
            meta=json.loads((out/'training_metadata.json').read_text());history=json.loads((out/'training_history.json').read_text())
            if meta['status']!='complete' or meta['smoke'] or meta['completed_updates']!=updates or len(history)!=updates:raise ValueError(f'Wrong training budget: {name}')
            if meta['clip_epsilon']!=eps or meta['kl_beta']!=beta:raise ValueError(f'Wrong ablation configuration: {name}')
            if meta['prompt_ids']!=[r['prompt_id'] for r in train[:updates*int(cfg['prompts_per_update'])]]:raise ValueError(f'Training prompt IDs differ: {name}')
            summary.update({'updates':updates,'epsilon':eps,'kl_beta':beta,'training_generated_tokens':meta['generated_tokens'],
                            'wall_seconds':meta['wall_seconds'],'peak_vram_gib':meta['peak_vram_bytes']/2**30,
                            'max_policy_gradient_norm':max(e['policy_gradient_norm'] for h in history for e in h['epochs']),
                            'max_value_gradient_norm':max(e['value_gradient_norm'] for h in history for e in h['epochs']),
                            'overflow_retries':sum(sum(e['overflow_retries'] for e in h['epochs']) for h in history)})
            histories.extend({'condition':name,**{k:v for k,v in h.items() if k!='epochs'}} for h in history)
        summaries.append(summary)
        for r in records:
            review.append({'condition':name,**r,'manual_quality_notes':'','reward_quality_agree':None})
    clipping=json.loads((root/'cached_clipping'/'diagnostics.json').read_text())
    if clipping['smoke'] or [r['epsilon'] for r in clipping['results']]!=cfg['clip_values']:raise ValueError('Missing full cached clipping diagnostics')
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(histories).to_csv(root/'training_trajectories.csv',index=False)
    pd.DataFrame(clipping['results']).to_csv(root/'cached_clipping.csv',index=False)
    review_path=root/'qualitative_review.jsonl'
    old={(r['condition'],r['prompt_id']):r for r in load_records(review_path)}
    for r in review:
        previous=old.get((r['condition'],r['prompt_id']),{})
        for k in ('manual_quality_notes','reward_quality_agree'):r[k]=previous.get(k,r[k])
    review_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    checks={'all_required_runs_complete':True,'matched_update_budgets':True,'fixed_training_and_evaluation_ids':True,
            'cached_epsilon_diagnostics_complete':True,'raw_generations_available':True}
    write_json(root/'completion_checks.json',checks);print(json.dumps(checks,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');a=p.parse_args();export(a.config)
if __name__=='__main__':main()

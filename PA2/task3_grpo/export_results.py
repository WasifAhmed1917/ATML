"""Budget/identity checks and numeric evidence; qualitative judgments remain manual."""
import argparse
import json
import pandas as pd
from common.data import load_yaml,read_jsonl
from task3_grpo.runtime import artifact_path,write_json,load_records,file_hash


def check_training(meta,history,cfg,train,updates,loss_type):
    if meta['status']!='complete' or meta['smoke'] or meta['completed_updates']!=updates or len(history)!=updates:
        raise ValueError('Incomplete full GRPO training budget')
    if [h['update'] for h in history]!=list(range(1,updates+1)):raise ValueError('Nonconsecutive updates')
    if meta['loss_type']!=loss_type or meta['num_generations']!=4:raise ValueError('Wrong normalization/K')
    count=updates*int(cfg['prompts_per_update'])
    if meta['prompt_ids']!=[r['prompt_id'] for r in train[:count]]:raise ValueError('Wrong training prompt IDs')
    if meta['midpoint_policy']!=cfg['paths']['grpo_midpoint_policy']:raise ValueError('Wrong starting midpoint')
    expected=count*4*int(cfg['max_completion_length'])
    if meta['max_generated_token_budget']!=expected or meta['generated_tokens']>expected:raise ValueError('Wrong generation budget')
    for key in ('seed','clip_epsilon','kl_beta','learning_rate','policy_epochs','max_prompt_length','max_completion_length',
                'mask_truncated_completions','generation','lora','prompts_per_update'):
        if meta['config'][key]!=cfg[key]:raise ValueError(f'Unmatched setting: {key}')


def export(config='configs/grpo.yaml'):
    cfg=load_yaml(config);root=artifact_path(cfg['results_dir']);train=read_jsonl(cfg['paths']['rl_prompt_train']);evaluation=read_jsonl(cfg['paths']['rl_prompt_eval'])
    expected={'standard':(20,'grpo'),'norm_grpo':(8,'grpo'),'norm_dr_grpo':(8,'dr_grpo')}
    summaries=[];histories=[];lengths=[];review=[];forks=[]
    for name in ['midpoint',*expected]:
        folder=root/name;em=json.loads((folder/'evaluation_metadata.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        records=load_records(folder/'generations.jsonl')
        if em['status']!='complete' or em['smoke'] or len(records)!=len(evaluation):raise ValueError(f'Incomplete evaluation: {name}')
        if [r['prompt_id'] for r in records]!=[r['prompt_id'] for r in evaluation]:raise ValueError(f'Evaluation IDs differ: {name}')
        if em['config']['eval_max_completion_length']!=cfg['eval_max_completion_length']:raise ValueError('Different evaluation cap')
        if name in expected:
            updates,loss_type=expected[name];out=artifact_path(f'outputs/task3_grpo/{name}')
            meta=json.loads((out/'training_metadata.json').read_text());history=json.loads((out/'training_history.json').read_text())
            check_training(meta,history,cfg,train,updates,loss_type)
            if name.startswith('norm_'):forks.append(meta)
            summary.update({'updates':updates,'loss_type':loss_type,'training_generated_tokens':meta['generated_tokens'],
                'optimization_updates':meta['optimization_updates'],'skipped_all_truncated_updates':updates-meta['optimization_updates'],
                'wall_seconds':meta['wall_seconds'],'peak_vram_gib':meta['peak_vram_bytes']/2**30,
                'max_gradient_norm':max(e['gradient_norm'] for h in history for e in h['epochs']),
                'overflow_retries':sum(e['overflow_retries'] for h in history for e in h['epochs'])})
            for h in history:
                histories.append({'condition':name,**{k:v for k,v in h.items() if k not in ('epochs','groups')}})
                for e in h['epochs']:
                    for r in e['length_statistics']:
                        lengths.append({'condition':name,'update':h['update'],'policy_epoch':e['policy_epoch'],**r})
                saved=json.loads((out/'rollouts'/f"update_{h['update']:03d}.json").read_text())
                if len(saved)!=int(cfg['prompts_per_update'])*4:raise ValueError('Missing grouped rollout records')
                if sum(r['response_length'] for r in saved)!=h['generated_tokens']:raise ValueError('Rollout token counts differ')
                if any(r['truncated'] and r['training_tokens'] for r in saved):raise ValueError('Truncated completion used for training')
                if not (out/'rollouts'/f"update_{h['update']:03d}.pt").exists():raise ValueError('Missing rollout tensors')
        summaries.append(summary)
        review.extend({'condition':name,**r,'manual_quality_notes':'','reward_quality_agree':None} for r in records)
    # All configuration differences between the two forks must be limited to the explicit loss type/run name.
    if forks[0]['config']!=forks[1]['config'] or forks[0]['prompt_ids']!=forks[1]['prompt_ids']:
        raise ValueError('Normalization forks are not matched')
    cached=json.loads((root/'group_sizes'/'diagnostics.json').read_text())
    if cached['cache_sha256']!=file_hash(cfg['group_cache']):raise ValueError('Cached analysis uses another asset')
    for bin_name in ('all','lower_reward','higher_reward'):
        selected=[r for r in cached['summary'] if r['difficulty_bin']==bin_name]
        if sorted(r['k'] for r in selected)!=[2,4,8] or len({r['total_generations'] for r in selected})!=1:
            raise ValueError('Group study does not have equal generation budgets within bins')
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(histories).to_csv(root/'training_trajectories.csv',index=False)
    columns=['condition','update','policy_epoch','completion_index','response_length','length_bin','advantage',
             'policy_logp_abs_gradient_sum','policy_logp_abs_gradient_mean','normalization_denominator']
    frame=pd.DataFrame(lengths,columns=columns);frame.to_csv(root/'length_gradient_statistics.csv',index=False)
    if not frame.empty:
        frame.groupby(['condition','length_bin']).agg(n_completions=('response_length','size'),
            response_length_mean=('response_length','mean'),abs_advantage_mean=('advantage',lambda v:v.abs().mean()),
            policy_logp_abs_gradient_mean=('policy_logp_abs_gradient_mean','mean'),
            policy_logp_abs_gradient_sum_mean=('policy_logp_abs_gradient_sum','mean')).reset_index().to_csv(root/'length_conditioned_summary.csv',index=False)
    review_path=root/'qualitative_review.jsonl';old={(r['condition'],r['prompt_id']):r for r in load_records(review_path)}
    for r in review:
        for key in ('manual_quality_notes','reward_quality_agree'):r[key]=old.get((r['condition'],r['prompt_id']),{}).get(key,r[key])
    review_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    checks={'all_required_runs_complete':True,'matched_normalization_forks':True,'fixed_prompt_ids':True,
        'equal_cached_generation_budgets':True,'fixed_difficulty_bins':True,'truncated_completions_masked':True,
        'raw_generations_and_rollouts_available':True,'length_statistics_exported':True}
    write_json(root/'completion_checks.json',checks);print(json.dumps(checks,indent=2));return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args();export(a.config)
if __name__=='__main__':main()

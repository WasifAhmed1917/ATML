"""Partition every cached prompt's eight completions into equal-generation K groups."""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from common.data import load_yaml,read_jsonl
from task3_grpo.runtime import artifact_path,write_json,file_hash


def validate_cache(rows):
    groups=defaultdict(list)
    for r in rows:
        if not np.isfinite(r['reward']):raise ValueError('Non-finite cached reward')
        groups[r['prompt_id']].append(r)
    for prompt,records in groups.items():
        if sorted(r['generation_index'] for r in records)!=list(range(8)):
            raise ValueError(f'Expected exactly eight ordered completions per prompt: {prompt}')
        if len({r['source_index'] for r in records})!=1:raise ValueError('Inconsistent cached source identity')
    if not groups:raise ValueError('Empty group cache')
    return {p:sorted(rs,key=lambda r:r['generation_index']) for p,rs in sorted(groups.items())}


def analyze_cache(rows,group_sizes=(2,4,8),eps=1e-6):
    prompts=validate_cache(rows)
    ranked=sorted(prompts,key=lambda p:(np.mean([r['reward'] for r in prompts[p]]),p))
    split=len(ranked)//2
    bins={p:('lower_reward' if i<split else 'higher_reward') for i,p in enumerate(ranked)}
    definitions=[{'prompt_id':p,'difficulty_bin':bins[p],'cached_mean_reward':float(np.mean([r['reward'] for r in prompts[p]]))} for p in ranked]
    details=[]
    for k in group_sizes:
        if k not in (2,4,8):raise ValueError('K must divide the eight-completion cache: 2, 4, or 8')
        for prompt,records in prompts.items():
            for start in range(0,8,k):
                group=records[start:start+k];reward=np.asarray([r['reward'] for r in group],dtype=float)
                centered=reward-reward.mean();std=float(reward.std());adv=centered/(std+eps)
                eligible=np.asarray([not r['clipped_at_max'] for r in group],dtype=bool)
                details.append({'k':k,'prompt_id':prompt,'difficulty_bin':bins[prompt],'group_index':start//k,
                    'n_generations':k,'generation_indices':[r['generation_index'] for r in group],
                    'reward_std':std,'informative':std>eps,'centered_reward_variance':float(centered.var()),
                    'advantage_variance':float(adv.var()),'eligible_completions':int(eligible.sum()),
                    'effective_informative':bool((np.abs(adv[eligible])>eps).any()),
                    'masked_advantage_second_moment':float(np.mean((adv*eligible)**2))})
    summaries=[]
    for k in group_sizes:
        for bin_name in ('all','lower_reward','higher_reward'):
            selected=[r for r in details if r['k']==k and (bin_name=='all' or r['difficulty_bin']==bin_name)]
            if not selected:continue
            summaries.append({'k':k,'difficulty_bin':bin_name,'n_groups':len(selected),
                'n_prompts':len({r['prompt_id'] for r in selected}),'total_generations':sum(r['n_generations'] for r in selected),
                'informative_group_rate':float(np.mean([r['informative'] for r in selected])),
                'group_reward_std_mean':float(np.mean([r['reward_std'] for r in selected])),
                'group_reward_std_std':float(np.std([r['reward_std'] for r in selected])),
                'group_relative_signal_variance_mean':float(np.mean([r['advantage_variance'] for r in selected])),
                'centered_reward_variance_mean':float(np.mean([r['centered_reward_variance'] for r in selected])),
                'effective_informative_group_rate':float(np.mean([r['effective_informative'] for r in selected])),
                'eligible_completion_fraction':sum(r['eligible_completions'] for r in selected)/sum(r['n_generations'] for r in selected),
                'masked_advantage_second_moment_mean':float(np.mean([r['masked_advantage_second_moment'] for r in selected]))})
    return {'partition':'all eight completions per prompt, ordered by generation_index, consecutive disjoint groups of K',
        'difficulty_definition':'median rank of fixed full-eight cached mean learned reward; proxy, not ground-truth difficulty',
        'informative_threshold':eps,'difficulty_assignments':definitions,'summary':summaries,'groups':details}


def run(config='configs/grpo.yaml'):
    cfg=load_yaml(config);rows=read_jsonl(cfg['group_cache']);result=analyze_cache(rows,cfg['group_sizes'])
    result['cache_sha256']=file_hash(cfg['group_cache']);out=artifact_path(cfg['results_dir'])/'group_sizes'
    write_json(out/'diagnostics.json',result)
    pd.DataFrame(result['summary']).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(result['groups']).to_csv(out/'groups.csv',index=False)
    pd.DataFrame(result['difficulty_assignments']).to_csv(out/'difficulty_bins.csv',index=False)
    print(pd.DataFrame(result['summary']).to_string(index=False),flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args();run(a.config)
if __name__=='__main__':main()

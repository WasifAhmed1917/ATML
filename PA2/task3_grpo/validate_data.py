"""Verify released prompt pools, cached groups, and GRPO midpoint identity."""
import argparse
from common.data import load_yaml,read_jsonl,repo_path
from task3_grpo.analyze_groups import validate_cache
from task3_grpo.runtime import artifact_path,write_json,file_hash


def validate(config='configs/grpo.yaml'):
    cfg=load_yaml(config);train=read_jsonl(cfg['paths']['rl_prompt_train']);evaluation=read_jsonl(cfg['paths']['rl_prompt_eval'])
    if len(train)!=1200 or len(evaluation)!=200:raise ValueError('Expected 1200 training and 200 held-out prompts')
    for rows in (train,evaluation):
        if len({r['prompt_id'] for r in rows})!=len(rows):raise ValueError('Duplicate prompt IDs')
    if {r['prompt_id'] for r in train}&{r['prompt_id'] for r in evaluation}:raise ValueError('Prompt pools overlap')
    cache=read_jsonl(cfg['group_cache']);groups=validate_cache(cache)
    if len(groups)!=24 or len(cache)!=192:raise ValueError('Expected 24 cached prompts with eight completions each')
    lookup={r['prompt_id']:r for r in train+evaluation}
    for r in cache:
        if r['prompt_id'] not in lookup or lookup[r['prompt_id']]['source_index']!=r['source_index']:raise ValueError('Cached identity mismatch')
        if not 0<int(r['completion_tokens'])<=int(cfg['cache_generation_cap']):raise ValueError('Cached token count outside released cap')
        if r['clipped_at_max'] and (r['terminated_with_eos'] or int(r['completion_tokens'])!=int(cfg['cache_generation_cap'])):
            raise ValueError('Cached truncation flags contradict EOS/length')
    midpoint=repo_path(cfg['paths']['grpo_midpoint_policy'])
    for name in ('adapter_config.json','adapter_model.safetensors'):
        if not (midpoint/name).exists():raise FileNotFoundError(midpoint/name)
    manifest={'n_train_prompts':len(train),'n_eval_prompts':len(evaluation),'n_cached_prompts':len(groups),'n_cached_generations':len(cache),
        'cache_sha256':file_hash(cfg['group_cache']),'train_sha256':file_hash(cfg['paths']['rl_prompt_train']),
        'eval_sha256':file_hash(cfg['paths']['rl_prompt_eval']),
        'midpoint_sha256':file_hash(midpoint/'adapter_model.safetensors'),
        'standard_train_prompt_ids':[r['prompt_id'] for r in train[:cfg['updates']*cfg['prompts_per_update']]],
        'fork_train_prompt_ids':[r['prompt_id'] for r in train[:cfg['fork_updates']*cfg['prompts_per_update']]],
        'eval_prompt_ids':[r['prompt_id'] for r in evaluation]}
    write_json(artifact_path(cfg['results_dir'])/'data_manifest.json',manifest)
    print('GRPO assets validated: 1200 train / 200 held-out / 24 cached prompts x 8 completions.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args();validate(a.config)
if __name__=='__main__':main()

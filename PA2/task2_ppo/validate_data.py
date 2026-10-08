"""Check the released PPO prompt identities, midpoint files, and exact cache token alignment."""
import argparse
from common.data import load_yaml,read_jsonl,repo_path
from common.models import load_tokenizer
from task2_ppo.analyze_clipping import load_cached_rollouts,reconstruct_cached_row
from task2_ppo.runtime import artifact_path,write_json,file_hash


def validate(config='configs/ppo.yaml'):
    cfg=load_yaml(config);train=read_jsonl(cfg['paths']['rl_prompt_train']);evaluation=read_jsonl(cfg['paths']['rl_prompt_eval'])
    if len(train)!=1200 or len(evaluation)!=200:raise ValueError('Expected released 1200 training and 200 held-out prompts')
    for rows in (train,evaluation):
        if len({r['prompt_id'] for r in rows})!=len(rows):raise ValueError('Duplicate prompt IDs')
    if {r['prompt_id'] for r in train}&{r['prompt_id'] for r in evaluation}:raise ValueError('Training/evaluation prompt overlap')
    for base,files in [(cfg['paths']['ppo_midpoint_policy'],['adapter_config.json','adapter_model.safetensors']),
                       (cfg['paths']['ppo_midpoint_value'],['config.json','model.safetensors'])]:
        for f in files:
            if not (repo_path(base)/f).exists():raise FileNotFoundError(repo_path(base)/f)
    tok=load_tokenizer(cfg['base_model']);tok.truncation_side='right'
    cache=load_cached_rollouts(cfg['cached_rollouts']);lookup={r['prompt_id']:r for r in train+evaluation}
    if len(cache)!=32:raise ValueError('Expected all 32 supplied diagnostic rollouts')
    for row in cache:reconstruct_cached_row(tok,row,lookup[row['prompt_id']],max_prompt_length=cfg['max_prompt_length'])
    manifest={'n_train_prompts':len(train),'n_eval_prompts':len(evaluation),'n_cached_rollouts':len(cache),
              'cache_tokens_exactly_reconstructed':True,'cache_sha256':file_hash(cfg['cached_rollouts']),
              'train_sha256':file_hash(cfg['paths']['rl_prompt_train']),'eval_sha256':file_hash(cfg['paths']['rl_prompt_eval']),
              'standard_train_prompt_ids':[r['prompt_id'] for r in train[:cfg['updates']*cfg['prompts_per_update']]],
              'fork_train_prompt_ids':[r['prompt_id'] for r in train[:cfg['fork_updates']*cfg['prompts_per_update']]],
              'eval_prompt_ids':[r['prompt_id'] for r in evaluation]}
    write_json(artifact_path(cfg['results_dir'])/'data_manifest.json',manifest)
    print('PPO assets validated: 1200 train / 200 held-out / 32 cached rollouts; exact cached response-token alignment.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');a=p.parse_args();validate(a.config)
if __name__=='__main__':main()

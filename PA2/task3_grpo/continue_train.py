"""Released GRPO midpoint continuation with per-update recovery checkpoints."""
from __future__ import annotations
import argparse
import json
import time
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from common.data import load_yaml,read_jsonl
from common.logging_utils import set_seed
from common.models import load_policy,load_reward_model,load_tokenizer,trainable_parameters
from common.metrics import masked_mean
from task3_grpo.runtime import (artifact_path,write_json,signature,provenance,rng_state,restore_rng,
    atomic_torch_save,parameter_state,load_parameter_state,collect_group_rollout,update_on_rollout,
    group_diagnostics,rollout_records)


def prepare_grpo_continuation(cfg):
    set_seed(int(cfg['seed']));tok=load_tokenizer(cfg['base_model']);tok.truncation_side='right'
    policy=load_policy(cfg,adapter_path=cfg['paths']['grpo_midpoint_policy'],trainable=True)
    for p in policy.parameters():
        if p.requires_grad:p.data=p.data.float()
    policy.eval() # Match old/new probabilities; dropout off but gradients enabled during replay.
    rm,rt=load_reward_model(cfg);rt.truncation_side='right'
    return {'cfg':cfg,'tokenizer':tok,'policy':policy,'reward_model':rm,'reward_tokenizer':rt,
            'optimizer':AdamW(trainable_parameters(policy),lr=float(cfg['learning_rate']))}


def run_grpo(config='configs/grpo.yaml',output=None,updates=None,loss_type='grpo',run_name='standard',
             resume=False,skip_completed=False,smoke=False):
    if not torch.cuda.is_available() and not smoke:raise RuntimeError('Full GRPO runs require a CUDA GPU')
    cfg=load_yaml(config)
    if updates is not None:cfg['updates']=int(updates)
    if smoke:
        cfg['updates']=1;cfg['max_completion_length']=16
        # Test-only: exercise FP16 backward even when all 16-token samples hit the cap.
        # Full experiments retain the required whole-completion truncation mask.
        cfg['mask_truncated_completions']=False
    rows=read_jsonl(cfg['paths']['rl_prompt_train']);count=int(cfg['updates'])*int(cfg['prompts_per_update'])
    if count>len(rows):raise ValueError('Insufficient distinct training prompts')
    selected=rows[:count];out=artifact_path(output or cfg['output']);out.mkdir(parents=True,exist_ok=True)
    sig=signature(cfg,selected,loss_type=loss_type,run_name=run_name,smoke=smoke)
    metadata_path=out/'training_metadata.json';checkpoint_path=out/'continuation.pt'
    if metadata_path.exists():
        previous=json.loads(metadata_path.read_text())
        if previous['signature']!=sig:raise ValueError('GRPO saved code/config/IDs differ')
        if previous['status']=='complete' and skip_completed:
            print(f'Skipping completed GRPO run: {run_name}',flush=True);return previous
        if not resume:raise ValueError('GRPO run exists: pass --resume')
    meta={**provenance(cfg),'signature':sig,'run_name':run_name,'status':'running','smoke':smoke,
          'updates':cfg['updates'],'loss_type':loss_type,'num_generations':cfg['num_generations'],
          'prompt_ids':[r['prompt_id'] for r in selected],'max_generated_token_budget':count*int(cfg['num_generations'])*int(cfg['max_completion_length']),
          'midpoint_policy':cfg['paths']['grpo_midpoint_policy'],'reference':'base policy with LoRA disabled',
          'probability_convention':'raw teacher-forced logp; dropout disabled during rollout/replay',
          'advantage_convention':'population reward std + 1e-6 within each prompt; include truncated rewards before masking',
          'normalization_scope':'policy surrogate only; released KL term remains globally token normalized'}
    write_json(metadata_path,meta);bundle=prepare_grpo_continuation(cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available() and cfg['dtype'] in ('float16','fp16'))
    history=[];start_update=0;prior_seconds=0.;peak=0
    if checkpoint_path.exists():
        if not resume:raise ValueError('Checkpoint exists: pass --resume')
        cp=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if cp['signature']!=sig:raise ValueError('GRPO checkpoint signature differs')
        load_parameter_state(bundle['policy'],cp['policy']);bundle['optimizer'].load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        restore_rng(cp['rng']);history=cp['history'];start_update=cp['completed_updates'];prior_seconds=cp['wall_seconds'];peak=cp['peak_vram_bytes']
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    def save_checkpoint(completed):
        atomic_torch_save(checkpoint_path,{'signature':sig,'completed_updates':completed,'policy':parameter_state(bundle['policy']),
            'optimizer':bundle['optimizer'].state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'history':history,
            'wall_seconds':prior_seconds+time.perf_counter()-started,
            'peak_vram_bytes':max(peak,torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
        write_json(out/'training_history.json',history)
    if not checkpoint_path.exists():save_checkpoint(0)
    for update in tqdm(range(start_update,int(cfg['updates'])),initial=start_update,total=int(cfg['updates']),desc=f'GRPO: {run_name}'):
        tick=time.perf_counter();seed=int(cfg['seed'])+update;set_seed(seed)
        bs=int(cfg['prompts_per_update']);rollout=collect_group_rollout(bundle,selected[update*bs:(update+1)*bs])
        diagnostics=update_on_rollout(bundle,rollout,scaler,loss_type);mask=rollout['response_mask'];tokens=int(mask.sum())
        record={'update':update+1,'generation_seed':seed,'reward':float(rollout['rewards'].mean()),
            'sampled_kl_log_ratio':float(masked_mean(rollout['old_logprobs']-rollout['ref_logprobs'],mask)),
            'rollout_entropy':float(masked_mean(rollout['old_entropy'],mask)),
            'response_length':float(mask.sum(-1).mean()),'generated_tokens':tokens,
            'eligible_training_tokens':int(rollout['training_mask'].sum()),'truncated_fraction':sum(rollout['truncated'])/len(rollout['truncated']),
            **group_diagnostics(rollout),'epochs':diagnostics,'wall_seconds':time.perf_counter()-tick}
        for key in ('loss','policy_term','sampled_kl','clip_fraction','entropy','gradient_norm','overflow_retries','loss_scale'):
            record[key]=sum(d[key] for d in diagnostics)/len(diagnostics)
        record['skipped_all_truncated']=all(d['skipped_all_truncated'] for d in diagnostics)
        history.append(record)
        atomic_torch_save(out/'rollouts'/f'update_{update+1:03d}.pt',{k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in rollout.items()})
        write_json(out/'rollouts'/f'update_{update+1:03d}.json',rollout_records(rollout,update+1,seed));save_checkpoint(update+1)
        print(f"{run_name}: {update+1}/{cfg['updates']}, reward={record['reward']:.3f}, group std={record['group_reward_std']:.3f}, eligible tokens={record['eligible_training_tokens']}",flush=True)
    bundle['policy'].save_pretrained(out);bundle['tokenizer'].save_pretrained(out)
    meta.update({'status':'complete','completed_updates':len(history),'generated_tokens':sum(h['generated_tokens'] for h in history),
        'optimization_updates':sum(not h['skipped_all_truncated'] for h in history),
        'wall_seconds':prior_seconds+time.perf_counter()-started,
        'peak_vram_bytes':max(peak,torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
    write_json(metadata_path,meta);print(json.dumps(meta,indent=2),flush=True);return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');p.add_argument('--output');p.add_argument('--updates',type=int)
    p.add_argument('--loss-type',choices=['grpo','dr_grpo'],default='grpo');p.add_argument('--run-name',default='standard')
    for f in ('resume','skip-completed','smoke'):p.add_argument('--'+f,action='store_true')
    a=p.parse_args();run_grpo(a.config,a.output,a.updates,a.loss_type,a.run_name,a.resume,a.skip_completed,a.smoke)
if __name__=='__main__':main()

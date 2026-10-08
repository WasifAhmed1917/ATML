"""Resumable PPO continuation from the exact supplied policy and critic midpoint."""
from __future__ import annotations
import argparse
import json
import time
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from common.data import load_yaml,read_jsonl,repo_path
from common.logging_utils import set_seed
from common.models import (load_policy,load_value_model,load_reward_model,load_tokenizer,
                           trainable_parameters,value_parameter_groups)
from common.metrics import masked_mean
from task2_ppo.ppo import compute_gae,shaped_rewards,normalize_advantages,ppo_policy_loss,value_mse_loss,active_clip_fraction
from task2_ppo.runtime import (artifact_path,write_json,provenance,signature,rng_state,restore_rng,
    atomic_torch_save,parameter_state,load_parameter_state,collect_rollout,token_statistics,
    response_values,joint_update,cpu_rollout,rollout_records)


def prepare_ppo_continuation(cfg):
    set_seed(int(cfg['seed']))
    tokenizer=load_tokenizer(cfg['base_model']); tokenizer.truncation_side='right'
    policy=load_policy(cfg,adapter_path=cfg['paths']['ppo_midpoint_policy'],trainable=True)
    critic=load_value_model(cfg,cfg['paths']['ppo_midpoint_value'],train_mode=cfg['value_train_mode'])
    # FP16 frozen bases with FP32 optimizer parameters (including the critic scalar head).
    for model in (policy,critic):
        for parameter in model.parameters():
            if parameter.requires_grad: parameter.data=parameter.data.float()
    # PPO replay uses the same deterministic network as rollout collection. Gradients remain enabled.
    policy.eval(); critic.eval()
    if hasattr(critic,'gradient_checkpointing_enable'):
        critic.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    rm,rt=load_reward_model(cfg)
    return {'cfg':cfg,'tokenizer':tokenizer,'policy':policy,'value_model':critic,'reward_model':rm,
            'reward_tokenizer':rt,
            'policy_optimizer':AdamW(trainable_parameters(policy),lr=float(cfg['policy_learning_rate'])),
            'value_optimizer':AdamW(value_parameter_groups(critic,float(cfg['value_lora_learning_rate']),
                                                       float(cfg['value_head_learning_rate'])),weight_decay=0.)}


def update_on_rollout(bundle, rollout, scaler):
    cfg=bundle['cfg']; policy=bundle['policy']; critic=bundle['value_model']; mask=rollout['response_mask']
    rewards=shaped_rewards(rollout['effective_terminal_reward'],rollout['old_logprobs'],rollout['ref_logprobs'],mask,cfg['kl_beta'])
    advantages,returns=compute_gae(rewards,rollout['values'],mask,cfg['gamma'],cfg['gae_lambda'])
    advantages=normalize_advantages(advantages,mask).detach(); returns=returns.detach()
    rollout.update({'shaped_rewards':rewards,'advantages':advantages,'returns':returns})
    diagnostics=[]
    for epoch in range(int(cfg['ppo_epochs'])):
        def backward():
            logp,entropy=token_statistics(policy,rollout,entropy=True)
            pl,ratio,clip=ppo_policy_loss(logp,rollout['old_logprobs'],advantages,mask,cfg['clip_epsilon'])
            values=response_values(critic,rollout)
            vl=value_mse_loss(values,returns,mask)
            total=pl+float(cfg['value_coef'])*vl
            if not torch.isfinite(total): raise FloatingPointError('Non-finite PPO loss')
            scaler.scale(total).backward()
            return {'policy_loss':float(pl.detach()),'value_loss':float(vl.detach()),
                    'entropy':float(masked_mean(entropy.detach(),mask)),
                    'clip_fraction':float(clip),'active_clip_fraction':float(active_clip_fraction(ratio,advantages,mask,cfg['clip_epsilon'])),
                    'approx_old_policy_kl':float(masked_mean((ratio-1)-(logp.detach()-rollout['old_logprobs']),mask))}
        d,norms,retries,scale=joint_update([policy,critic],[bundle['policy_optimizer'],bundle['value_optimizer']],
                                           scaler,backward,float(cfg['max_grad_norm']))
        diagnostics.append({**d,'policy_gradient_norm':norms[0],'value_gradient_norm':norms[1],
                            'overflow_retries':retries,'loss_scale':scale,'ppo_epoch':epoch+1})
    return diagnostics


def run_ppo(config='configs/ppo.yaml',output=None,updates=None,clip_epsilon=None,kl_beta=None,
            run_name='standard',resume=False,skip_completed=False,smoke=False):
    if not torch.cuda.is_available() and not smoke: raise RuntimeError('Full PPO runs require a CUDA GPU')
    cfg=load_yaml(config)
    if updates is not None: cfg['updates']=int(updates)
    if clip_epsilon is not None: cfg['clip_epsilon']=float(clip_epsilon)
    if kl_beta is not None: cfg['kl_beta']=float(kl_beta)
    if smoke: cfg['updates']=1; cfg['max_response_length']=16
    out=artifact_path(output or cfg['output']); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(cfg['paths']['rl_prompt_train'])
    count=int(cfg['updates'])*int(cfg['prompts_per_update'])
    if count>len(rows): raise ValueError('Not enough distinct training prompts for fixed budget')
    selected=rows[:count] # Same first prompts for every midpoint fork; no condition-dependent sampling.
    sig=signature(cfg,selected,run_name=run_name,smoke=smoke)
    meta_path=out/'training_metadata.json'; checkpoint_path=out/'continuation.pt'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']!=sig: raise ValueError('Saved PPO run has different code/config/prompt IDs')
        if meta['status']=='complete' and skip_completed:
            print(f'Skipping completed PPO run: {run_name}',flush=True); return meta
        if not resume: raise ValueError('PPO artifacts exist: pass --resume')
    meta={**provenance(cfg),'signature':sig,'run_name':run_name,'smoke':smoke,'status':'running',
          'updates':int(cfg['updates']),'clip_epsilon':float(cfg['clip_epsilon']),'kl_beta':float(cfg['kl_beta']),
          'prompt_ids':[r.get('prompt_id') for r in selected],'source_indices':[r.get('source_index') for r in selected],
          'max_generated_token_budget':count*int(cfg['max_response_length']),
          'dropout_during_ppo':'disabled through eval mode; gradients enabled',
          'reference':'original base policy with adapters disabled; raw teacher-forced log probabilities',
          'midpoint_policy':cfg['paths']['ppo_midpoint_policy'],'midpoint_value':cfg['paths']['ppo_midpoint_value']}
    write_json(meta_path,meta)
    bundle=prepare_ppo_continuation(cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available() and cfg['dtype'] in ('float16','fp16'))
    history=[];start_update=0; prior_seconds=0.; peak=0
    if checkpoint_path.exists():
        if not resume: raise ValueError('Checkpoint exists: pass --resume')
        cp=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if cp['signature']!=sig: raise ValueError('Checkpoint signature differs')
        load_parameter_state(bundle['policy'],cp['policy']); load_parameter_state(bundle['value_model'],cp['critic'])
        bundle['policy_optimizer'].load_state_dict(cp['policy_optimizer']);bundle['value_optimizer'].load_state_dict(cp['value_optimizer'])
        scaler.load_state_dict(cp['scaler']); restore_rng(cp['rng'])
        history=cp['history']; start_update=cp['completed_updates'];prior_seconds=cp['wall_seconds'];peak=cp['peak_vram_bytes']
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    started=time.perf_counter()
    def save_checkpoint(completed):
        atomic_torch_save(checkpoint_path,{'signature':sig,'completed_updates':completed,
            'policy':parameter_state(bundle['policy']),'critic':parameter_state(bundle['value_model']),
            'policy_optimizer':bundle['policy_optimizer'].state_dict(),'value_optimizer':bundle['value_optimizer'].state_dict(),
            'scaler':scaler.state_dict(),'rng':rng_state(),'history':history,
            'wall_seconds':prior_seconds+time.perf_counter()-started,
            'peak_vram_bytes':max(peak,torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
        write_json(out/'training_history.json',history)
    if not checkpoint_path.exists(): save_checkpoint(0)
    for update in tqdm(range(start_update,int(cfg['updates'])),initial=start_update,total=int(cfg['updates']),desc=f'PPO: {run_name}'):
        tick=time.perf_counter(); seed=int(cfg['seed'])+update;set_seed(seed)
        bs=int(cfg['prompts_per_update']); batch=selected[update*bs:(update+1)*bs]
        rollout=collect_rollout(bundle,batch)
        diagnostics=update_on_rollout(bundle,rollout,scaler)
        mask=rollout['response_mask']; tokens=int(mask.sum())
        record={'update':update+1,'generation_seed':seed,
                'reward':float(rollout['raw_terminal_reward'].mean()),
                'effective_reward':float(rollout['effective_terminal_reward'].mean()),
                'sampled_kl_token_mean':float(masked_mean(rollout['old_logprobs']-rollout['ref_logprobs'],mask)),
                'response_length':float(mask.sum(-1).mean()),'generated_tokens':tokens,
                'terminated_fraction':sum(rollout['terminated_with_eos'])/bs,
                'truncated_fraction':sum(rollout['truncated'])/bs,'epochs':diagnostics,
                'wall_seconds':time.perf_counter()-tick}
        for key in diagnostics[0]:
            if key!='ppo_epoch': record[key]=sum(d[key] for d in diagnostics)/len(diagnostics)
        history.append(record)
        atomic_torch_save(out/'rollouts'/f'update_{update+1:03d}.pt',cpu_rollout(rollout))
        write_json(out/'rollouts'/f'update_{update+1:03d}.json',rollout_records(rollout,update+1,seed))
        save_checkpoint(update+1)
        print(f"{run_name}: update {update+1}/{cfg['updates']}, reward={record['reward']:.3f}, KL={record['sampled_kl_token_mean']:.5f}, tokens={tokens}",flush=True)
    bundle['policy'].save_pretrained(out);bundle['tokenizer'].save_pretrained(out)
    bundle['value_model'].save_pretrained(out/'critic_adapter')
    meta.update({'status':'complete','completed_updates':len(history),'generated_tokens':sum(r['generated_tokens'] for r in history),
                 'wall_seconds':prior_seconds+time.perf_counter()-started,
                 'peak_vram_bytes':max(peak,torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
    write_json(meta_path,meta); print(json.dumps(meta,indent=2),flush=True);return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--output');p.add_argument('--updates',type=int)
    p.add_argument('--clip-epsilon',type=float);p.add_argument('--kl-beta',type=float);p.add_argument('--run-name',default='standard')
    for f in ('resume','skip-completed','smoke'): p.add_argument('--'+f,action='store_true')
    a=p.parse_args();run_ppo(a.config,a.output,a.updates,a.clip_epsilon,a.kl_beta,a.run_name,a.resume,a.skip_completed,a.smoke)
if __name__=='__main__':main()

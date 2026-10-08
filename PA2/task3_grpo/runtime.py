"""GRPO provenance, group rollouts, masking, and bounded-memory optimization."""
from __future__ import annotations
import hashlib
import torch
from common.data import repo_path,prompt_messages
from common.generation import batch_generate,score_reward_pairs
from common.models import reference_mode
from common.metrics import masked_mean
from task2_ppo.runtime import (artifact_path,write_json,append_record,load_records,object_hash,file_hash,
    stats,atomic_torch_save,rng_state,restore_rng,parameter_state,load_parameter_state,token_statistics,
    provenance as shared_provenance)
from task3_grpo.grpo import group_relative_advantages,grpo_policy_loss,mask_truncated_sequences


def code_hash():
    h=hashlib.sha256()
    for folder in ('task3_grpo','task2_ppo','common'):
        for p in sorted(repo_path(folder).glob('*.py')):
            h.update(str(p.relative_to(repo_path('.'))).encode());h.update(p.read_bytes())
    return h.hexdigest()


def provenance(cfg,**extra):
    result=shared_provenance(cfg,**extra);result['code_sha256']=code_hash();return result


def signature(cfg,rows,**extra):
    return object_hash({'config':cfg,'rows':rows,'code':code_hash(),
                       'asset_revision':'0b350481fb03f5525a35bcdec4131bd4fe487f98',**extra})


def joint_update(models, optimizers, scaler, backward, max_grad_norm=1., max_retries=16):
    """Replay the identical GRPO minibatch after AMP overflow; neither optimizer partially steps."""
    state=rng_state()
    for attempt in range(max_retries+1):
        restore_rng(state)
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        diagnostics=backward()
        for opt in optimizers: scaler.unscale_(opt)
        norms=[torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],max_grad_norm) for m in models]
        if all(torch.isfinite(n) for n in norms):
            scale=float(scaler.get_scale())
            for opt in optimizers: scaler.step(opt)
            scaler.update()
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            return diagnostics,[float(n) for n in norms],attempt,scale
        if not scaler.is_enabled(): raise FloatingPointError('Non-finite GRPO gradients')
        old=float(scaler.get_scale()); scaler.update(new_scale=old*scaler.get_backoff_factor())
        print(f'FP16 overflow: replaying SAME GRPO epoch; scale {old:g} -> {scaler.get_scale():g}',flush=True)
    for opt in optimizers: opt.zero_grad(set_to_none=True)
    raise FloatingPointError('GRPO gradients remained non-finite after bounded recovery')


def length_bin(length):
    return 'short_1_127' if length<128 else ('medium_128_255' if length<256 else 'long_256_plus')


@torch.no_grad()
def collect_group_rollout(bundle,rows):
    cfg=bundle['cfg'];model=bundle['policy'];tok=bundle['tokenizer'];k=int(cfg['num_generations'])
    repeated=[r for r in rows for _ in range(k)];messages=[prompt_messages(r) for r in repeated]
    model.eval();was_cache=model.config.use_cache;model.config.use_cache=True
    try:
        output=batch_generate(model,tok,messages,int(cfg['max_prompt_length']),int(cfg['max_completion_length']),**cfg['generation'])
    finally:model.config.use_cache=was_cache
    output={key:value.clone() if torch.is_tensor(value) else value for key,value in output.items()}
    output['attention_mask'][:,output['prompt_width']:]=output['response_mask'].long()
    # Forward one completion at a time: rollout tensors retain all groups, vocabulary logits do not.
    old=[];ref=[];entropy=[]
    for j in range(len(repeated)):
        one=select_completion(output,j)
        lp,en=token_statistics(model,one,entropy=True)
        with reference_mode(model):reference,_=token_statistics(model,one)
        old.append(lp);ref.append(reference);entropy.append(en)
    rewards=score_reward_pairs(bundle['reward_model'],bundle['reward_tokenizer'],messages,output['responses'],
                               max_length=int(cfg['reward_max_length']))
    group_ids=torch.arange(len(rows),device=rewards.device).repeat_interleave(k)
    advantages=group_relative_advantages(rewards,group_ids)
    training_mask=mask_truncated_sequences(output['response_mask'],output['truncated']) if cfg['mask_truncated_completions'] else output['response_mask'].clone()
    return {**output,'old_logprobs':torch.cat(old),'ref_logprobs':torch.cat(ref),'old_entropy':torch.cat(entropy),
            'rewards':rewards,'group_ids':group_ids,'advantages':advantages,'training_mask':training_mask,
            'rows':repeated,'messages':messages}


def select_completion(rollout,index):
    keys=('sequences','attention_mask','response_ids','response_mask')
    return {**{key:rollout[key][index:index+1] for key in keys},'prompt_width':rollout['prompt_width']}


def update_on_rollout(bundle,rollout,scaler,loss_type='grpo'):
    cfg=bundle['cfg'];policy=bundle['policy'];n=rollout['response_ids'].shape[0]
    total_tokens=float(rollout['training_mask'].sum());epochs=[]
    if not total_tokens:
        # No response is eligible; do not turn AdamW decay into an unintended learning update.
        return [{'loss':0.,'policy_term':0.,'sampled_kl':0.,'clip_fraction':0.,'entropy':0.,
                 'gradient_norm':0.,'overflow_retries':0,'loss_scale':float(scaler.get_scale()),
                 'skipped_all_truncated':True,'length_statistics':[]}]
    for epoch in range(int(cfg['policy_epochs'])):
        def backward():
            sums={'loss':0.,'policy_term':0.,'sampled_kl':0.,'clip_fraction':0.,'entropy':0.};lengths=[]
            for j in range(n):
                one=select_completion(rollout,j);mask=rollout['training_mask'][j:j+1];valid=float(mask.sum())
                if not valid:continue
                new,entropy=token_statistics(policy,one,entropy=True)
                old=rollout['old_logprobs'][j:j+1];ref=rollout['ref_logprobs'][j:j+1];adv=rollout['advantages'][j:j+1]
                policy_term,diag=grpo_policy_loss(new,old,adv,mask,ref,float(cfg['clip_epsilon']),0.,loss_type,int(cfg['max_completion_length']))
                logratio=ref-new;kl=((logratio.exp()-logratio-1.)*mask).sum()/total_tokens
                normalized_policy=policy_term/n
                loss=normalized_policy+float(cfg['kl_beta'])*kl
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite GRPO objective')
                # Exact policy-surrogate sensitivity to each sampled logp, excluding KL.
                # This directly exposes the 1/T versus 1/L normalization effect.
                sensitivity=torch.autograd.grad(normalized_policy,new,retain_graph=True)[0].detach().abs()*mask
                count=int(mask.sum());lengths.append({'completion_index':j,'response_length':count,
                    'length_bin':length_bin(count),'advantage':float(adv[0]),
                    'policy_logp_abs_gradient_sum':float(sensitivity.sum()),
                    'policy_logp_abs_gradient_mean':float(sensitivity.sum()/count),
                    'normalization_denominator':count if loss_type=='grpo' else int(cfg['max_completion_length'])})
                scaler.scale(loss).backward()
                sums['loss']+=float(loss.detach());sums['policy_term']+=float(normalized_policy.detach());sums['sampled_kl']+=float(kl.detach())
                for key,value in [('clip_fraction',diag['clip_fraction']),('entropy',masked_mean(entropy.detach(),mask))]:
                    sums[key]+=float(value)*valid/total_tokens
            return {**sums,'length_statistics':lengths}
        d,norms,retries,scale=joint_update([policy],[bundle['optimizer']],scaler,backward,float(cfg['max_grad_norm']))
        epochs.append({**d,'gradient_norm':norms[0],'overflow_retries':retries,'loss_scale':scale,
                       'skipped_all_truncated':False,'policy_epoch':epoch+1})
    return epochs


def group_diagnostics(rollout):
    groups=[]
    for group in torch.unique(rollout['group_ids']):
        selected=rollout['group_ids']==group;rewards=rollout['rewards'][selected]
        std=float(rewards.std(unbiased=False));a=rollout['advantages'][selected]
        eligible=rollout['training_mask'][selected].sum(-1)>0
        groups.append({'group_id':int(group),'reward_std':std,'uninformative':std<=1e-6,
                       'eligible_completions':int(eligible.sum()),
                       'effective_signal':bool(((a.abs()>1e-6)&eligible).any())})
    return {'groups':groups,'group_reward_std':sum(g['reward_std'] for g in groups)/len(groups),
            'uninformative_group_fraction':sum(g['uninformative'] for g in groups)/len(groups),
            'effective_informative_group_fraction':sum(g['effective_signal'] for g in groups)/len(groups)}


def rollout_records(rollout,update,seed):
    records=[]
    for j,row in enumerate(rollout['rows']):
        n=int(rollout['response_mask'][j].sum());valid=int(rollout['training_mask'][j].sum())
        records.append({'update':update,'group_id':int(rollout['group_ids'][j]),'generation_index':j,
             'generation_seed':seed,'prompt_id':row['prompt_id'],'source_index':row['source_index'],
             'messages':rollout['messages'][j],'response':rollout['responses'][j],
             'response_token_ids':rollout['response_ids'][j,:n].cpu().tolist(),'response_length':n,
             'training_tokens':valid,'reward_score':float(rollout['rewards'][j]),'advantage':float(rollout['advantages'][j]),
             'terminated_with_eos':rollout['terminated_with_eos'][j],'truncated':rollout['truncated'][j],
             'masked_from_training':valid==0})
    return records

"""PPO artifacts, token alignment, and atomic mixed-precision updates."""
from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common.data import repo_path, prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import reference_mode
from task1_dpo.utils import artifact_path, write_json, append_record, load_records, object_hash, file_hash, stats, provenance as shared_provenance


def code_hash():
    h = hashlib.sha256()
    for p in sorted(repo_path('task2_ppo').glob('*.py')) + sorted(repo_path('common').glob('*.py')):
        h.update(str(p.relative_to(repo_path('.'))).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def provenance(cfg, **extra):
    result=shared_provenance(cfg, **extra)
    result["code_sha256"]=code_hash()
    return result


def signature(cfg, rows, **extra):
    return object_hash({'config':cfg, 'rows':rows, 'code':code_hash(),
                        'asset_revision':'0b350481fb03f5525a35bcdec4131bd4fe487f98', **extra})


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch_save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); torch.save(value,tmp); os.replace(tmp,path)


def parameter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_parameter_state(model, state):
    target={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(target)!=set(state): raise ValueError('Trainable checkpoint parameter names differ')
    with torch.no_grad():
        for n,p in target.items(): p.copy_(state[n].to(p.device,p.dtype))


def token_statistics(model, rollout, entropy=False, chunk_size=16):
    """Raw-policy log probabilities and exact categorical entropy at pre-action positions."""
    width=rollout['prompt_width']; response=rollout['response_ids']; mask=rollout['response_mask']
    logits=model(input_ids=rollout['sequences'],attention_mask=rollout['attention_mask'],
                 use_cache=False,return_dict=True).logits[:,width-1:width-1+response.shape[1]]
    chosen=[]; ent=[]
    def calc(x,y):
        lp=F.log_softmax(x.float(),dim=-1)
        selected=lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)
        e=-(lp.exp()*lp).sum(-1) if entropy else selected.new_zeros(selected.shape)
        return selected,e
    for start in range(0,response.shape[1],chunk_size):
        x=logits[:,start:start+chunk_size]; y=response[:,start:start+chunk_size]
        a,b=checkpoint(calc,x,y,use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else calc(x,y)
        chosen.append(a); ent.append(b)
    return torch.cat(chosen,dim=1)*mask, torch.cat(ent,dim=1)*mask


def response_values(model, rollout):
    width=rollout['prompt_width']; steps=rollout['response_ids'].shape[1]
    core=model.get_base_model() if hasattr(model,'get_base_model') else model
    backbone=getattr(core,core.base_model_prefix)
    hidden=backbone(input_ids=rollout['sequences'],attention_mask=rollout['attention_mask'],
                    output_hidden_states=True,return_dict=True,use_cache=False).hidden_states[-1]
    head=core.score if hasattr(core,'score') else core.classifier
    # PEFT keeps modules_to_save heads in the base dtype; FP16 gradients cannot be unscaled.
    # Continuation promotes trainable heads to FP32, so explicitly cast their hidden inputs.
    dtype=next((p.dtype for p in head.parameters() if p.requires_grad),next(head.parameters()).dtype)
    return head(hidden[:,width-1:width-1+steps].to(dtype)).squeeze(-1).float()


@torch.no_grad()
def collect_rollout(bundle, rows, max_response_length=None):
    cfg=bundle['cfg']; tok=bundle['tokenizer']; policy=bundle['policy']
    messages=[prompt_messages(r) for r in rows]
    policy.eval(); bundle['value_model'].eval()
    tok.truncation_side='right'
    was_cache=policy.config.use_cache
    policy.config.use_cache=True
    try:
        output=batch_generate(policy,tok,messages,int(cfg['max_prompt_length']),
                              int(max_response_length or cfg['max_response_length']),**cfg['generation'])
    finally:
        policy.config.use_cache=was_cache
    # Inference tensors must be cloned outside inference_mode before gradient-bearing replay.
    output={k:v.clone() if torch.is_tensor(v) else v for k,v in output.items()}
    output['attention_mask'][:,output['prompt_width']:]=output['response_mask'].long()
    old,entropy=token_statistics(policy,output,entropy=True)
    with reference_mode(policy): ref,_=token_statistics(policy,output)
    values=response_values(bundle['value_model'],output)
    raw=score_reward_pairs(bundle['reward_model'],bundle['reward_tokenizer'],messages,output['responses'],
                           max_length=int(cfg['reward_max_length']))
    effective=raw-torch.tensor([0. if eos else float(cfg['missing_eos_penalty']) for eos in output['terminated_with_eos']],device=raw.device)
    return {**output,'old_logprobs':old,'ref_logprobs':ref,'values':values,
            'raw_terminal_reward':raw,'effective_terminal_reward':effective,'old_entropy':entropy,
            'rows':rows,'messages':messages}


def joint_update(models, optimizers, scaler, backward, max_grad_norm=1., max_retries=16):
    """Replay the identical PPO minibatch after AMP overflow; neither optimizer partially steps."""
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
        if not scaler.is_enabled(): raise FloatingPointError('Non-finite PPO gradients')
        old=float(scaler.get_scale()); scaler.update(new_scale=old*scaler.get_backoff_factor())
        print(f'FP16 overflow: replaying SAME PPO epoch; scale {old:g} -> {scaler.get_scale():g}',flush=True)
    for opt in optimizers: opt.zero_grad(set_to_none=True)
    raise FloatingPointError('PPO gradients remained non-finite after bounded recovery')


def cpu_rollout(rollout):
    return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in rollout.items()}


def rollout_records(rollout, update, seed):
    result=[]
    for j,row in enumerate(rollout['rows']):
        n=int(rollout['response_mask'][j].sum()); tok=rollout['response_ids'][j,:n].cpu().tolist()
        result.append({'update':update,'generation_seed':seed,'source_index':row.get('source_index'),
                       'prompt_id':row.get('prompt_id'),'messages':rollout['messages'][j],
                       'response':rollout['responses'][j],'response_token_ids':tok,'response_length':n,
                       'terminated_with_eos':rollout['terminated_with_eos'][j],'truncated':rollout['truncated'][j],
                       'raw_terminal_reward':float(rollout['raw_terminal_reward'][j]),
                       'effective_terminal_reward':float(rollout['effective_terminal_reward'][j])})
    return result

"""Reconstruct the supplied cached batch without changing its reward, old logp, or values."""
from __future__ import annotations
import argparse
import json
import torch
from tqdm.auto import tqdm
from common.data import load_yaml,repo_path,read_jsonl,prompt_messages
from common.models import load_policy,load_tokenizer
from common.logging_utils import set_seed
from common.metrics import masked_mean
from task2_ppo.ppo import shaped_rewards,compute_gae,normalize_advantages,ppo_policy_loss,active_clip_fraction
from task2_ppo.runtime import artifact_path,write_json,atomic_torch_save,token_statistics,signature,file_hash


def load_cached_rollouts(path):
    rows=torch.load(repo_path(path),map_location='cpu',weights_only=False)
    if not isinstance(rows,list) or not rows:raise ValueError('Expected nonempty PPO cache list')
    for row in rows:
        if 'old_logprobs' not in row:row['old_logprobs']=row['old_policy_logprobs']
        if 'ref_logprobs' not in row:row['ref_logprobs']=row['reference_logprobs']
        for k in ('source_index','response','values','effective_terminal_reward','old_logprobs','ref_logprobs'):
            if k not in row:raise ValueError(f'PPO cache missing {k}')
        n=int(row['response_tokens'])
        if any(len(row[k])!=n for k in ('old_logprobs','ref_logprobs','values')):raise ValueError('Cached token lengths differ')
    return rows


def reconstruct_cached_row(tokenizer,cached,prompt_row,device='cpu',max_prompt_length=256):
    messages=prompt_messages(prompt_row)
    prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)[:max_prompt_length]
    ids=tokenizer.encode(cached['response'],add_special_tokens=False)
    if cached['terminated_with_eos']:ids.append(tokenizer.eos_token_id)
    if len(ids)!=int(cached['response_tokens']):
        raise ValueError('Cached response cannot be reconstructed exactly: token count mismatch. Do not align by padding/truncating old logp.')
    response=torch.tensor([ids],device=device);seq=torch.tensor([prompt+ids],device=device)
    return {'sequences':seq,'attention_mask':torch.ones_like(seq),'prompt_width':len(prompt),
            'response_ids':response,'response_mask':torch.ones_like(response,dtype=torch.float32)}


def analyze(config='configs/ppo.yaml',smoke=False):
    cfg=load_yaml(config);cache=load_cached_rollouts(cfg['cached_rollouts']);
    if smoke:cache=cache[:2]
    prompts=read_jsonl(cfg['paths']['rl_prompt_eval'])+read_jsonl(cfg['paths']['rl_prompt_train'])
    lookup={r['prompt_id']:r for r in prompts}
    set_seed(int(cfg['seed']));tok=load_tokenizer(cfg['base_model']);tok.truncation_side='right'
    model=load_policy(cfg,adapter_path=cfg['paths']['ppo_midpoint_policy'],trainable=False);device=next(model.parameters()).device
    batches=[]
    with torch.no_grad():
        for row in tqdm(cache,desc='Cached PPO ratios'):
            prompt=lookup[row['prompt_id']]
            if prompt['source_index']!=row['source_index']:raise ValueError('Cached source ID mismatch')
            batch=reconstruct_cached_row(tok,row,prompt,device,int(cfg['max_prompt_length']))
            new,_=token_statistics(model,batch)
            old=row['old_logprobs'].to(device).float()[None];ref=row['ref_logprobs'].to(device).float()[None]
            values=row['values'].to(device).float()[None];mask=batch['response_mask']
            rewards=shaped_rewards(torch.tensor([row['effective_terminal_reward']],device=device),old,ref,mask,cfg['kl_beta'])
            advantages,returns=compute_gae(rewards,values,mask,cfg['gamma'],cfg['gae_lambda'])
            batches.append({'prompt_id':row['prompt_id'],'new_logprobs':new.cpu(),'old_logprobs':old.cpu(),
                            'advantages':advantages.cpu(),'returns':returns.cpu(),'mask':mask.cpu()})
    # Normalize once across the fixed batch, not separately for each epsilon or response.
    new=torch.cat([r['new_logprobs'].flatten() for r in batches])[None]
    old=torch.cat([r['old_logprobs'].flatten() for r in batches])[None]
    mask=torch.ones_like(new);advantages=normalize_advantages(torch.cat([r['advantages'].flatten() for r in batches])[None],mask)
    results=[]
    for eps in cfg['clip_values']:
        loss,ratio,affected=ppo_policy_loss(new,old,advantages,mask,eps)
        results.append({'epsilon':float(eps),'policy_loss':float(loss),'clipped_surrogate':-float(loss),
                        'unclipped_surrogate':float(masked_mean(ratio*advantages,mask)),
                        'affected_token_fraction':float(affected),
                        'active_clip_fraction':float(active_clip_fraction(ratio,advantages,mask,eps)),
                        'n_tokens':int(mask.sum()),'n_rollouts':len(cache)})
    out=artifact_path(cfg['results_dir'])/('cached_clipping_smoke' if smoke else 'cached_clipping');out.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(out/'reconstructed_batch.pt',{'rows':batches,'normalized_advantages':advantages,'ratios':torch.exp(new-old)})
    write_json(out/'diagnostics.json',{'signature':signature(cfg,[r['prompt_id'] for r in cache],cache_sha256=file_hash(cfg['cached_rollouts'])),
         'smoke':smoke,'ratios_source':'supplied midpoint replay versus supplied cached old-policy logp',
         'advantage_normalization':'all valid cached tokens, once, shared across epsilon values','results':results})
    print(json.dumps(results,indent=2),flush=True);return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--smoke',action='store_true');a=p.parse_args();analyze(a.config,a.smoke)
if __name__=='__main__':main()

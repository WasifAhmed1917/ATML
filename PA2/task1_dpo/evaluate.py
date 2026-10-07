"""Fixed held-out DPO, sampled token KL, reward, and length evaluation."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from common.data import (load_yaml, read_jsonl, prompt_messages_from_preference, prompt_messages,
                         repo_path)
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import sampled_kl, word_count, parse_word_limit, word_limit_compliance
from common.models import load_policy, load_tokenizer, load_reward_model, reference_mode
from task1_dpo.encoding import overflow_mode,pair_diagnostics
from task1_dpo.train import make_collate
from task1_dpo.utils import (artifact_path, result_dir, identity, file_hash, object_hash, provenance,
                            write_json, append_record, load_records, stats, sequence_logprobs,
                            token_logprobs, to_device, resolved_model_revision)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def adapter_metadata(adapter):
    if adapter is None: return {'run_name':'sft','beta':None,'train_examples':0,'epochs':0,'smoke':False}
    path=Path(adapter)/'training_metadata.json'
    if not path.exists(): raise FileNotFoundError(f'Missing completed training metadata: {path}')
    meta=json.loads(path.read_text())
    if meta.get('status')!='complete': raise ValueError('Adapter training is not complete')
    return meta


def validate_existing_records(records, rows):
    if len(records)>len(rows): raise ValueError('Evaluation has more records than fixed data')
    for i,r in enumerate(records):
        if r['row_index']!=i or r.get('prompt_id')!=rows[i].get('prompt_id'):
            raise ValueError('Saved evaluation IDs do not match the fixed row order')


@torch.no_grad()
def evaluate_pairs(model, tokenizer, rows, cfg, beta, path):
    records=load_records(path); validate_existing_records(records,rows)
    collate=make_collate(tokenizer,int(cfg['max_sequence_length']))
    bs=int(cfg['batch_size'])
    for start in tqdm(range(len(records),len(rows),bs),desc=f'Preference pairs: {path.name}'):
        selected=rows[start:start+bs]
        chosen,rejected=collate(selected)
        chosen=to_device(chosen,model); rejected=to_device(rejected,model)
        with reference_mode(model):
            rc=sequence_logprobs(model,chosen)[0]; rr=sequence_logprobs(model,rejected)[0]
        pc=sequence_logprobs(model,chosen)[0]; pr=sequence_logprobs(model,rejected)[0]
        margin=(pc-rc)-(pr-rr); losses=-F.logsigmoid(beta*margin)
        for j,row in enumerate(selected):
            record={**identity(row,start+j), **pair_diagnostics(tokenizer,row,int(cfg['max_sequence_length'])), 'policy_chosen_logp':float(pc[j]),
                    'policy_rejected_logp':float(pr[j]),'ref_chosen_logp':float(rc[j]),
                    'ref_rejected_logp':float(rr[j]),'preference_margin':float(margin[j]),
                    'dpo_loss':float(losses[j]),'preference_correct':bool(margin[j]>0),
                    'chosen_scored_tokens':int(chosen['response_mask'][j].sum()),
                    'rejected_scored_tokens':int(rejected['response_mask'][j].sum())}
            if not all(__import__('math').isfinite(record[k]) for k in
                       ('dpo_loss','preference_margin','policy_chosen_logp','policy_rejected_logp')):
                raise FloatingPointError('Non-finite held-out preference metric')
            append_record(path,record); records.append(record)
    return records


@torch.no_grad()
def evaluate_generations(model,tokenizer,rows,cfg,path,preference_prompts=True):
    records=load_records(path); validate_existing_records(records,rows)
    for index in tqdm(range(len(records),len(rows)),desc=f'Generate + KL: {path.name}'):
        row=rows[index]
        messages=prompt_messages_from_preference(row) if preference_prompts else prompt_messages(row)
        # Per-row seeds make resume and condition comparisons independent of earlier RNG use.
        seed=int(cfg['seed'])+index
        set_seed(seed)
        output=batch_generate(model,tokenizer,[messages],
                   max_prompt_length=int(cfg['max_sequence_length']),
                   max_new_tokens=int(cfg['max_generation_tokens']),**cfg['generation'])
        width=output['prompt_width']; ids=output['sequences']; attn=output['attention_mask']
        # Generation helper marks post-EOS positions in response_mask; apply it to attention too.
        mask=output['response_mask']; attn[:,width:]=mask.long()
        pc=token_logprobs(model,ids,attn)[:,width-1:]
        with reference_mode(model): rr=token_logprobs(model,ids,attn)[:,width-1:]
        kl=float(sampled_kl(pc,rr,mask)); count=int(mask.sum())
        prompt_text='\n'.join(str(m['content']) for m in messages)
        text=output['responses'][0]
        prompt_tokens=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        record={**identity(row,index),'messages':messages,'generation_seed':seed,'response':text,
                'response_token_ids':output['response_ids'][0,:count].cpu().tolist(),
                'response_length':count,'response_words':word_count(text),
                'sampled_kl_token_mean':kl,'sampled_kl_sum':kl*count,
                'terminated_with_eos':output['terminated_with_eos'][0],
                'truncated':output['truncated'][0],'prompt_tokens_before_truncation':prompt_tokens,
                'prompt_truncated':prompt_tokens>int(cfg['max_sequence_length']),
                'word_limit':parse_word_limit(prompt_text),
                'word_limit_compliant':word_limit_compliance(prompt_text,text)}
        append_record(path,record); records.append(record)
    return records


@torch.no_grad()
def score_saved_generations(cfg,groups,paths):
    if all(len(load_records(path))==len(rows) for rows,path in zip(groups,paths)): return
    rm,tok=load_reward_model(cfg)
    for rows,path in zip(groups,paths):
        previous=load_records(path); validate_existing_records(previous,rows)
        bs=int(cfg['batch_size'])
        for start in tqdm(range(len(previous),len(rows),bs),desc=f'Reward: {path.name}'):
            batch=rows[start:start+bs]
            scores=score_reward_pairs(rm,tok,[r['messages'] for r in batch],
                                      [r['response'] for r in batch],max_length=1024)
            for j,row in enumerate(batch):
                text=tok.apply_chat_template(row['messages']+[{'role':'assistant','content':row['response']}],
                                             tokenize=False,add_generation_prompt=False)
                tokens=len(tok(text,add_special_tokens=True)['input_ids'])
                append_record(path,{**row,'reward_score':float(scores[j]),
                                    'reward_input_tokens':tokens,'reward_input_truncated':tokens>1024})
    del rm,tok; clear_memory()


def preference_summary(records):
    return {'n_pairs':len(records),'heldout_dpo_loss':stats([r['dpo_loss'] for r in records])['mean'],
            'preference_accuracy':sum(r['preference_correct'] for r in records)/len(records),
            'preference_tie_rate':sum(r['preference_margin']==0 for r in records)/len(records)}


def generation_summary(records):
    length=stats([r['response_length'] for r in records]); reward=stats([r['reward_score'] for r in records])
    valid=[r for r in records if r['word_limit_compliant'] is not None]
    return {'n_prompts':len(records),
            'sampled_kl_token_mean':sum(r['sampled_kl_sum'] for r in records)/sum(r['response_length'] for r in records),
            'reward_score_mean':reward['mean'],'reward_score_std':reward['std'],
            **{'response_length_'+k:v for k,v in length.items()},
            'generation_truncated_fraction':sum(r['truncated'] for r in records)/len(records),
            'prompt_truncated_fraction':sum(r['prompt_truncated'] for r in records)/len(records),
            'reward_input_truncated_fraction':sum(r['reward_input_truncated'] for r in records)/len(records),
            'word_limit_n':len(valid),
            'word_limit_compliance':sum(r['word_limit_compliant'] for r in valid)/len(valid) if valid else None}


def run_evaluation(config_path='configs/dpo.yaml',adapter=None,name='standard',
                   length=False,resume=False,smoke=False):
    cfg=load_yaml(config_path)
    adapter=str(artifact_path(adapter)) if adapter else None
    meta=adapter_metadata(adapter)
    if bool(meta.get('smoke',False)) != bool(smoke) and adapter:
        raise ValueError('Smoke adapters cannot be used for full-course evaluation')
    if adapter and meta.get('prompt_overflow_mode','strict') != overflow_mode():
        raise ValueError('Evaluation overflow handling differs from training')
    if adapter and meta['config_sha256']!=object_hash(cfg):
        raise ValueError('Evaluation config differs from the training config')
    beta=float(meta['beta'] if meta['beta'] is not None else cfg['beta'])
    sources={'standard_pairs':cfg['paths']['dpo_standard_eval']}
    if length:
        sources['length_pairs']=cfg['paths']['dpo_length_eval']
        sources['word_limit']=cfg['paths']['word_limit_prompts']
    datasets={k:read_jsonl(p)[:2] if smoke else read_jsonl(p) for k,p in sources.items()}
    results=result_dir(cfg,name); results.mkdir(parents=True,exist_ok=True)
    spec=provenance(cfg,adapter=adapter,adapter_sha256=file_hash(Path(adapter)/'adapter_model.safetensors') if adapter else None,
                    name=name,beta=beta,smoke=smoke,length=length,prompt_overflow_mode=overflow_mode(),
                    data_sha256={k:file_hash(v) for k,v in sources.items()},
                    generation_batch_size=1,reward_batch_size=int(cfg['batch_size']),reward_max_length=1024,
                    kl_convention='sum of sampled logp differences / all valid response tokens, including EOS',
                    preference_convention='summed response logp, including EOS; strict margin > 0')
    signature=object_hash(spec)
    manifest=results/'evaluation_metadata.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['signature']!=signature: raise ValueError('Existing evaluation has different data/config/adapter/code')
        if not resume: raise FileExistsError('Evaluation exists; use --resume to reuse its exact saved records')
    write_json(manifest,{**spec,'signature':signature,'status':'running'})
    set_seed(int(cfg['seed']))
    tok=load_tokenizer(cfg['base_model']); model=load_policy(cfg,adapter_path=adapter,trainable=False)
    # Generation truncates only overlong prompts, retaining the assistant prefix.
    # Pair preprocessing has its own shared-context budgeting rule.
    tok.truncation_side = 'left'
    pairs=evaluate_pairs(model,tok,datasets['standard_pairs'],cfg,beta,results/'standard_pairs.jsonl')
    length_pairs=evaluate_pairs(model,tok,datasets['length_pairs'],cfg,beta,results/'length_pairs.jsonl') if length else None
    generations=evaluate_generations(model,tok,datasets['standard_pairs'],cfg,results/'generations_raw.jsonl')
    words=evaluate_generations(model,tok,datasets['word_limit'],cfg,results/'word_limit_raw.jsonl',False) if length else None
    model_revision=resolved_model_revision(model)
    del model,tok; clear_memory()
    groups=[generations]; paths=[results/'generations.jsonl']
    if length: groups.append(words); paths.append(results/'word_limit_generations.jsonl')
    score_saved_generations(cfg,groups,paths)
    scored=load_records(paths[0])
    summary={'condition':name,'beta':beta,'train_examples':meta['train_examples'],
             'epochs':meta['epochs'],'smoke':smoke,**preference_summary(pairs),**generation_summary(scored)}
    if length:
        summary['strata']={s:preference_summary([r for r in length_pairs if r['length_stratum']==s])
                           for s in sorted(set(r['length_stratum'] for r in length_pairs))}
        summary['word_limit']=generation_summary(load_records(paths[1]))
    write_json(results/'evaluation_summary.json',summary)
    write_json(manifest,{**spec,'signature':signature,'status':'complete','base_model_revision':model_revision})
    print(json.dumps(summary,indent=2)); return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/dpo.yaml'); p.add_argument('--adapter')
    p.add_argument('--name',default='standard'); p.add_argument('--length',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run_evaluation(a.config,a.adapter,a.name,a.length,a.resume,a.smoke)
if __name__=='__main__':main()

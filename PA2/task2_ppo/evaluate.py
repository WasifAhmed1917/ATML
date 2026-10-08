"""Fixed 200-prompt held-out PPO evaluation with per-response resume."""
from __future__ import annotations
import argparse
import json
import torch
from tqdm.auto import tqdm
from common.data import load_yaml,read_jsonl,prompt_messages
from common.generation import batch_generate,score_reward_pairs
from common.logging_utils import set_seed
from common.models import load_policy,load_tokenizer,load_reward_model,reference_mode
from common.metrics import masked_mean
from task2_ppo.runtime import (artifact_path,write_json,append_record,load_records,stats,signature,
                              token_statistics,provenance,file_hash)


def summarize(records):
    tokens=sum(r['response_length'] for r in records)
    reward=stats([r['reward_score'] for r in records]);length=stats([r['response_length'] for r in records])
    return {'n_prompts':len(records),'reward_score_mean':reward['mean'],'reward_score_std':reward['std'],
            'sampled_kl_token_mean':sum(r['sampled_kl_sum'] for r in records)/tokens,
            'entropy_token_mean':sum(r['entropy_sum'] for r in records)/tokens,
            **{'response_length_'+k:v for k,v in length.items()},
            'terminated_fraction':sum(r['terminated_with_eos'] for r in records)/len(records),
            'generation_truncated_fraction':sum(r['truncated'] for r in records)/len(records),
            'prompt_truncated_fraction':sum(r['prompt_truncated'] for r in records)/len(records),
            'reward_input_truncated_fraction':sum(r['reward_input_truncated'] for r in records)/len(records)}


def run_evaluation(config='configs/ppo.yaml',adapter=None,name='standard',resume=False,smoke=False):
    cfg=load_yaml(config); rows=read_jsonl(cfg['paths']['rl_prompt_eval']);
    if smoke: rows=rows[:2];cfg['eval_max_response_length']=16
    if not adapter: adapter=cfg['paths']['ppo_midpoint_policy']
    adapter=artifact_path(adapter) if name!='midpoint' else __import__('common.data',fromlist=['repo_path']).repo_path(adapter)
    if name!='midpoint':
        meta=json.loads((adapter/'training_metadata.json').read_text())
        if meta['status']!='complete':raise ValueError('PPO adapter is not complete')
        if bool(meta['smoke'])!=bool(smoke):raise ValueError('Smoke/full evaluation mismatch')
    out=artifact_path(cfg['results_dir'])/name;out.mkdir(parents=True,exist_ok=True)
    sig=signature(cfg,rows,adapter_sha256=file_hash(adapter/'adapter_model.safetensors'),smoke=smoke)
    metadata=out/'evaluation_metadata.json';path=out/'generations.jsonl'
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        if previous['signature']!=sig:raise ValueError('Evaluation code/config/adapter/IDs differ')
        if not resume:raise ValueError('Evaluation exists: pass --resume')
    write_json(metadata,{**provenance(cfg),'signature':sig,'condition':name,'smoke':smoke,'status':'running',
                         'n_prompts':len(rows),'prompt_ids':[r['prompt_id'] for r in rows]})
    records=load_records(path)
    if len(records)>len(rows) or any(r['row_index']!=i or r['prompt_id']!=rows[i]['prompt_id'] for i,r in enumerate(records)):
        raise ValueError('Saved evaluation IDs differ from fixed prompts')
    if len(records)<len(rows):
        tok=load_tokenizer(cfg['base_model']);tok.truncation_side='right'
        policy=load_policy(cfg,adapter_path=str(adapter),trainable=False);rm,rt=load_reward_model(cfg)
        rt.truncation_side='right'
        for i in tqdm(range(len(records),len(rows)),initial=len(records),total=len(rows),desc=f'PPO held-out: {name}'):
            set_seed(int(cfg['seed'])+i);messages=prompt_messages(rows[i])
            with torch.no_grad():
                output=batch_generate(policy,tok,[messages],int(cfg['max_prompt_length']),int(cfg['eval_max_response_length']),**cfg['generation'])
                output['attention_mask'][:,output['prompt_width']:]=output['response_mask'].long()
                lp,entropy=token_statistics(policy,output,entropy=True)
                with reference_mode(policy):ref,_=token_statistics(policy,output)
                mask=output['response_mask'];n=int(mask.sum());text=output['responses'][0]
                reward=float(score_reward_pairs(rm,rt,[messages],[text],max_length=int(cfg['reward_max_length']))[0])
                rmtext=rt.apply_chat_template(messages+[{'role':'assistant','content':text}],tokenize=False,add_generation_prompt=False)
                rmcount=len(rt(rmtext)['input_ids']);prompt_count=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
                record={'row_index':i,'prompt_id':rows[i]['prompt_id'],'source_index':rows[i]['source_index'],
                    'messages':messages,'generation_seed':int(cfg['seed'])+i,'response':text,
                    'response_token_ids':output['response_ids'][0,:n].cpu().tolist(),'response_length':n,
                    'reward_score':reward,'effective_reward':reward-(0 if output['terminated_with_eos'][0] else float(cfg['missing_eos_penalty'])),
                    'sampled_kl_sum':float(((lp-ref)*mask).sum()),'entropy_sum':float((entropy*mask).sum()),
                    'sampled_kl_token_mean':float(masked_mean(lp-ref,mask)),
                    'entropy_token_mean':float(masked_mean(entropy,mask)),
                    'terminated_with_eos':output['terminated_with_eos'][0],'truncated':output['truncated'][0],
                    'prompt_tokens_before_truncation':prompt_count,'prompt_truncated':prompt_count>int(cfg['max_prompt_length']),
                    'reward_input_tokens':rmcount,'reward_input_truncated':rmcount>int(cfg['reward_max_length'])}
            append_record(path,record);records.append(record)
    summary={'condition':name,**summarize(records)};write_json(out/'summary.json',summary)
    metadata_obj=json.loads(metadata.read_text());metadata_obj['status']='complete';write_json(metadata,metadata_obj)
    print(json.dumps(summary,indent=2),flush=True);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--adapter');p.add_argument('--name',default='standard')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run_evaluation(a.config,a.adapter,a.name,a.resume,a.smoke)
if __name__=='__main__':main()

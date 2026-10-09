"""Greedy responses from the four prescribed frozen policies, saved batch by batch."""
from __future__ import annotations
import argparse
import json
import pandas as pd
import torch
from tqdm.auto import tqdm
from common.data import load_yaml,repo_path
from common.generation import batch_generate
from common.models import load_policy,load_tokenizer
from common.logging_utils import set_seed
from task4_safety.runtime import (POLICIES,output_dir,validate_protocol,validate_adapter,resume_records,
                                append_record,finish)


def policy_specs(cfg):return {name:cfg['policies'][name] for name in POLICIES}


def load_xstest(cfg):
    df=pd.read_csv(repo_path(cfg['paths']['xstest']))
    required={'xstest_id','prompt','benchmark_class','type'}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():raise ValueError('Invalid XSTest columns or null values')
    if df.xstest_id.duplicated().any() or set(df.benchmark_class)!= {'SAFE','UNSAFE'}:raise ValueError('Invalid XSTest IDs/classes')
    return df


def generate_for_policy(cfg,policy_name,batch_size=4,resume=False,smoke=False):
    validate_protocol(cfg)
    if policy_name not in POLICIES or batch_size<1:raise ValueError('Invalid policy/batch size')
    adapter,sha=validate_adapter(cfg,policy_name)
    rows=load_xstest(cfg).to_dict('records')
    if smoke:rows=rows[:2]
    out=output_dir(cfg);out.mkdir(parents=True,exist_ok=True)
    path=out/f'generated_{policy_name}.jsonl';meta=out/f'generation_{policy_name}_metadata.json'
    records=resume_records(cfg,rows,path,meta,resume,'generation',policy=policy_name,adapter_sha256=sha,
                           smoke=smoke,batch_size=batch_size,decoding={'do_sample':False,'max_prompt_length':256,
                            'max_new_tokens':int(cfg['safety_max_new_tokens']),'temperature':0.,'top_p':1.})
    if len(records)<len(rows):
        set_seed(int(cfg['seed']));tok=load_tokenizer(cfg['base_model']);tok.truncation_side='right'
        model=load_policy(cfg,adapter_path=str(adapter) if adapter else None,trainable=False)
        model.eval();model.config.use_cache=True
        progress=tqdm(total=len(rows),initial=len(records),desc=f'Safety generate: {policy_name}',unit='prompt')
        while len(records)<len(rows):
            start=len(records)
            # Replay an interrupted batch with its original membership/width before appending missing rows.
            batch_start=(start//batch_size)*batch_size
            chunk=rows[batch_start:batch_start+batch_size]
            prompts=[[{'role':'user','content':str(r['prompt'])}] for r in chunk]
            gen=batch_generate(model,tok,prompts,max_prompt_length=256,max_new_tokens=int(cfg['safety_max_new_tokens']),
                               temperature=0.,top_p=1.,do_sample=False)
            for j,r in enumerate(chunk):
                index=batch_start+j
                if index<len(records):continue
                n=int(gen['response_lengths'][j]);count=len(tok.apply_chat_template(prompts[j],tokenize=True,add_generation_prompt=True))
                record={'row_index':index,'xstest_id':int(r['xstest_id']),'policy':policy_name,
                        'prompt':str(r['prompt']),'benchmark_class':str(r['benchmark_class']),'type':str(r['type']),
                        'response':gen['responses'][j],'response_tokens':n,
                        'response_token_ids':gen['response_ids'][j,:n].cpu().tolist(),
                        'terminated_with_eos':bool(gen['terminated_with_eos'][j]),'truncated':bool(gen['truncated'][j]),
                        'prompt_tokens_before_truncation':count,'prompt_truncated':count>256}
                append_record(path,record);records.append(record);progress.update(1)
        progress.close()
    finish(meta);print(f'{policy_name}: {len(records)} deterministic safety responses saved',flush=True)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');p.add_argument('--policy',choices=POLICIES)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=load_yaml(a.config)
    for name in ([a.policy] if a.policy else POLICIES):generate_for_policy(cfg,name,a.batch_size,a.resume,a.smoke)
if __name__=='__main__':main()

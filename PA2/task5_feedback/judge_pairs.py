"""Unchanged course judge, with recoverable cache and input-bound row progress."""
from __future__ import annotations
import argparse
import json
import time
from tqdm.auto import tqdm
from common.data import load_yaml
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.metrics import verifier_preference
from task5_feedback.runtime import (output_dir, load_records, load_dataset, validate_protocol, resume_records,
                                   append_record, finish, object_hash, write_json)


def restore_cache(cfg, out, path):
    # The supplied judge writes its whole cache non-atomically. Reconstruct every
    # durable label from our JSONL if that write was interrupted, preserving labels
    # without altering the rubric, orientation, generation or parser.
    try:
        cache = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(cache, dict): raise ValueError('Judge cache is not a mapping')
    except (ValueError, json.JSONDecodeError):
        path.replace(path.with_name(path.name+'.interrupted')); cache = {}
    stub = PairwiseAIJudge.__new__(PairwiseAIJudge); stub.cfg = cfg
    for p in sorted(out.glob('*_pairs.jsonl')):
        for row in load_records(p):
            key = stub._key(row['question'], row['a_response'], row['b_response'])
            cache[key] = row['judge_preference']
    write_json(path, cache)


def score_pairs(cfg, inputs, path, resume, kind, **extra):
    meta = path.with_suffix('.meta.json')
    records = resume_records(cfg, inputs, path, meta, resume, kind, **extra)
    if len(records) == len(inputs):
        finish(meta,records); print(f'Skipping completed judge: {path.name}',flush=True); return records
    cache_path = output_dir(cfg)/'judge_cache.json'
    restore_cache(cfg,output_dir(cfg),cache_path)
    judge = PairwiseAIJudge(cfg,cache_path)
    for i in tqdm(range(len(records),len(inputs)),initial=len(records),total=len(inputs),desc=f'Pairwise {path.stem}'):
        pair = inputs[i]; before = time.perf_counter()
        key = judge._key(pair['question'],pair['a_response'],pair['b_response'])
        cached = key in judge.cache
        preference = judge.compare(pair['question'],pair['a_response'],pair['b_response'])
        seconds = time.perf_counter()-before
        record = {**pair,'row_index':i,'input_sha256':object_hash(pair),'judge_preference':preference,
                  'verifier_preference':verifier_preference(pair['a_exact_reward'],pair['b_exact_reward']),
                  'judge_cache_key':key,'judge_cache_hit':cached,'judge_seconds':seconds,
                  'orientation_swapped':int(key[:8],16)%2==1}
        append_record(path,record); records.append(record)
    finish(meta,records)
    return records


def judge_math(config='configs/feedback.yaml', dataset='gsm', policy='rlaif', resume=False, smoke=False):
    cfg=load_yaml(config);validate_protocol(cfg);rows=load_dataset(cfg,dataset,smoke)
    out=output_dir(cfg); sources=[]
    for name in (policy,'sft'):
        path=out/f'{dataset}_{name}_responses.jsonl'
        meta=json.loads(path.with_suffix('.meta.json').read_text())
        rec=load_records(path)
        if meta['status']!='complete' or meta['smoke']!=smoke or len(rec)!=len(rows):
            raise ValueError(f'Generate complete matching responses first: {name}')
        # Verify config/code/adapter/input signatures even when a stage is skipped.
        from task5_feedback.runtime import adapter_identity
        _, sha=adapter_identity(cfg,name)
        from task5_feedback.generate_responses import generate
        generate(config,dataset,name,True,smoke)
        sources.append(rec)
    inputs=[]
    for i,(a,b) in enumerate(zip(*sources)):
        if a['prompt_id']!=b['prompt_id']:raise ValueError('Unmatched math pairs')
        inputs.append({'prompt_id':a['prompt_id'],'dataset':dataset,'policy':policy,'question':a['question'],
                       'a_response':a['response'],'b_response':b['response'],
                       'a_exact_reward':a['exact_reward'],'b_exact_reward':b['exact_reward']})
    return score_pairs(cfg,inputs,out/f'{dataset}_{policy}_pairs.jsonl',resume,'math_pairwise',
                       dataset=dataset,policy=policy,smoke=smoke,judge_new_tokens=4,judge_do_sample=False,
                       orientation='released hash-balanced A/B',unparseable_rule='released TIE fallback')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--dataset',choices=['gsm','transfer'],default='gsm');p.add_argument('--policy',choices=['rlvr','rlaif'],default='rlaif')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();judge_math(a.config,a.dataset,a.policy,a.resume,a.smoke)

if __name__=='__main__':main()

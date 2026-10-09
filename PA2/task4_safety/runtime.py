"""Fixed safety protocol, artifact identity, and disconnect recovery."""
from __future__ import annotations
import hashlib
import json
from common.data import load_yaml,repo_path
from task1_dpo.utils import artifact_path,write_json,append_record,load_records,object_hash,file_hash,provenance

POLICIES=('sft','dpo','ppo','grpo')
LABEL_ORDER=('SAFE_ANSWER','JUSTIFIED_REFUSAL','UNSAFE_COMPLIANCE','OVER_REFUSAL','AMBIGUOUS')


def output_dir(cfg):return artifact_path(cfg['results_dir'])/'task4_safety'


def code_hash():
    h=hashlib.sha256()
    for folder in ('task4_safety','common'):
        for p in sorted(repo_path(folder).glob('*.py')):
            h.update(str(p.relative_to(repo_path('.'))).encode());h.update(p.read_bytes())
    return h.hexdigest()


def signature(cfg,rows,**extra):
    return object_hash({'config':cfg,'rows':rows,'code_sha256':code_hash(),**extra})


def validate_protocol(cfg):
    released=load_yaml('configs/feedback.yaml')
    # Values mandated by the release; refuse silent substitutions or ablation adapters.
    expected={'base_model':'Qwen/Qwen2.5-1.5B-Instruct','ai_judge_model':'Qwen/Qwen2.5-3B-Instruct',
              'dtype':'float16','quantize_frozen_models':True,'safety_max_new_tokens':256,
              'judge_max_new_tokens':64,'judge_temperature':0.,'manual_audit_per_class':30,'seed':6304}
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError(f'Task 4 requires released {k}={v!r}')
    for name in POLICIES:
        want=None if name=='sft' else f'outputs/task{dict(dpo=1,ppo=2,grpo=3)[name]}_{name}/standard'
        if cfg['policies'].get(name)!=want:raise ValueError(f'Fixed policy path required: {name}={want}')
    return released


def validate_adapter(cfg,name):
    if name=='sft':return None,None
    path=artifact_path(cfg['policies'][name]);meta_path=path/'training_metadata.json'
    if not meta_path.exists():raise FileNotFoundError(f'Complete standard {name} training first: {meta_path}')
    meta=json.loads(meta_path.read_text())
    if meta.get('status')!='complete' or meta.get('smoke') or meta.get('run_name')!='standard':
        raise ValueError(f'{name} must be the completed, full standard policy')
    if name=='dpo':
        if meta.get('epochs')!=1 or meta.get('train_examples')!=1500 or meta.get('beta')!=0.1:
            raise ValueError('Standard DPO requires one epoch, 1500 examples, beta 0.1')
    elif meta.get('updates')!=20 or meta.get('completed_updates')!=20:
        raise ValueError(f'Standard {name} requires 20 completed rollout updates')
    if meta.get('config',{}).get('base_model')!=cfg['base_model']:raise ValueError('Policy base model differs')
    weights=path/'adapter_model.safetensors'
    if not weights.exists():raise FileNotFoundError(weights)
    adapter_cfg=json.loads((path/'adapter_config.json').read_text())
    if adapter_cfg.get('base_model_name_or_path')!=cfg['base_model']:raise ValueError('Adapter belongs to a different base')
    return path,file_hash(weights)


def resume_records(cfg,rows,path,meta_path,resume,kind,**extra):
    sig=signature(cfg,rows,kind=kind,**extra)
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        if previous['signature']!=sig:raise ValueError('Saved safety code/config/input identity differs')
        if not resume:raise ValueError('Safety outputs exist: use --resume')
    elif path.exists():raise ValueError('Orphaned safety output without metadata')
    records=load_records(path)
    if len(records)>len(rows):raise ValueError('Too many saved safety records')
    for i,r in enumerate(records):
        if r.get('row_index')!=i or r['xstest_id']!=rows[i]['xstest_id']:raise ValueError('Saved safety row order differs')
    write_json(meta_path,{**provenance(cfg),'code_sha256':code_hash(),'signature':sig,'kind':kind,
                         'status':'running','n_rows':len(rows),**extra})
    return records


def finish(meta_path):
    meta=json.loads(meta_path.read_text());meta['status']='complete';write_json(meta_path,meta)

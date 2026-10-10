"""Course entry point: generation, fixed pairwise judging and aggregation."""
from __future__ import annotations
import argparse
from common.data import load_yaml
from task2_ppo.run_all import launch
from task5_feedback.metrics import policy_summary,pair_summary
from task5_feedback.runtime import POLICIES,load_dataset,output_dir,require_complete,write_json,object_hash,adapter_identity,validate_protocol


def policy_specs(cfg):return {p:cfg['policies'][p] for p in POLICIES}
def dataset_path(cfg,dataset):return cfg['paths'][{'gsm':'gsm_eval','transfer':'math_transfer_eval'}[dataset]]


def aggregate(cfg,dataset):
    validate_protocol(cfg);rows=load_dataset(cfg,dataset);out=output_dir(cfg);summary={'dataset':dataset,'policies':{}}
    for name in POLICIES:
        path=out/f'{dataset}_{name}_responses.jsonl'
        records,meta=require_complete(path,path.with_suffix('.meta.json'),len(rows))
        if meta['config_sha256']!=object_hash(cfg):
            raise ValueError('Mismatched result configuration')
        _,sha=adapter_identity(cfg,name)
        if meta.get('adapter_sha256')!=sha:raise ValueError('Saved adapter identity differs from the course release')
        s=policy_summary(records)
        if name!='sft':
            path=out/f'{dataset}_{name}_pairs.jsonl'
            pairs,_=require_complete(path,path.with_suffix('.meta.json'),len(rows));s.update(pair_summary(pairs))
        else:
            s.update(ai_win_rate_vs_sft=None,ai_tie_rate_vs_sft=None,ai_loss_rate_vs_sft=None,
                     ai_tie_adjusted_win_rate_vs_sft=None) # no invented self-judge baseline
        summary['policies'][name]=s
    write_json(out/f'{dataset}_summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--dataset',choices=['gsm','transfer'],default='gsm');p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all','generate','judge','export'],default='all');a=p.parse_args()
    base=['--config',a.config,'--dataset',a.dataset];flags=['--resume'] if a.resume else []
    if a.stage in ('all','generate'):
        for name in POLICIES:launch('task5_feedback.generate_responses',base+['--policy',name,*flags])
    if a.stage in ('all','judge'):
        for name in ('rlvr','rlaif'):launch('task5_feedback.judge_pairs',base+['--policy',name,*flags])
    if a.stage in ('all','export'):print(aggregate(load_yaml(a.config),a.dataset),flush=True)

if __name__=='__main__':main()

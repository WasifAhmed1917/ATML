"""Complete Task 1 pipeline; separate processes release GPU memory between stages."""
import argparse
import os
import subprocess
import sys
from common.data import load_yaml, repo_path
from task1_dpo.ablate_beta import beta_name
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/dpo.yaml');p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all','standard','beta','length','export'],default='all')
    p.add_argument('--no-sft-baseline',action='store_true',help='Skip optional SFT generation comparison')
    a=p.parse_args();cfg=load_yaml(a.config)
    def launch(module,args):
        cmd=[sys.executable,'-m',module,'--config',a.config,*args]
        print('Running:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,cwd=repo_path('.'),check=True)
    train_resume=['--resume','--skip-completed'] if a.resume else []
    eval_resume=['--resume'] if a.resume else []
    if a.stage in ('all','standard'):
        launch('task1_dpo.train',['--run-name','standard',*train_resume])
        launch('task1_dpo.evaluate',['--adapter',cfg['standard_output'],'--name','standard','--length',*eval_resume])
        if not a.no_sft_baseline:
            launch('task1_dpo.evaluate',['--name','sft','--length',*eval_resume])
    if a.stage in ('all','beta'):
        for beta in cfg['betas']:
            name=beta_name(float(beta));output=str(Path(cfg['standard_output']).parent/name)
            launch('task1_dpo.train',['--run-name',name,'--output',output,'--beta',str(beta),
                   '--max-examples',str(cfg['short_ablation_examples']),*train_resume])
            launch('task1_dpo.evaluate',['--adapter',output,'--name',name,*eval_resume])
    if a.stage in ('all','length'):
        launch('task1_dpo.train',['--run-name','length_balanced','--dataset',cfg['paths']['dpo_length_train'],
                                '--output',cfg['length_output'],*train_resume])
        launch('task1_dpo.evaluate',['--adapter',cfg['length_output'],'--name','length_balanced','--length',*eval_resume])
    if a.stage in ('all','export'):
        launch('task1_dpo.export_results',['--require-complete'])
if __name__=='__main__':main()

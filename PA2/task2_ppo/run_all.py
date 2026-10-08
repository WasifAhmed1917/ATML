"""Sequential subprocess stages release GPU memory and stream tqdm in Colab."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from common.data import load_yaml,repo_path


def launch(module,args):
    cmd=[sys.executable,'-u','-m',module,*args]
    print('Running: '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=repo_path('.'),env={**os.environ,'PYTHONUNBUFFERED':'1','USE_TF':'0'},check=True)


def run_stage(stage,config='configs/ppo.yaml',resume=False):
    cfg=load_yaml(config);common=['--config',config];flags=['--resume'] if resume else []
    def train_eval(name,extra):
        adapter=f'outputs/task2_ppo/{name}'
        launch('task2_ppo.continue_train',common+['--run-name',name,'--output',adapter,'--skip-completed',*flags,*extra])
        launch('task2_ppo.evaluate',common+['--adapter',adapter,'--name',name,*flags])
    if stage in ('all','standard'):
        launch('task2_ppo.evaluate',common+['--name','midpoint',*flags])
        train_eval('standard',[])
    if stage in ('all','clipping'):
        launch('task2_ppo.analyze_clipping',common)
        for eps in cfg['clip_values']:
            train_eval(f'clip_{eps:.2f}',['--updates',str(cfg['fork_updates']),'--clip-epsilon',str(eps)])
    if stage in ('all','kl'):
        for beta in cfg['kl_values']:
            train_eval(f'kl_{beta:.2f}',['--updates',str(cfg['fork_updates']),'--kl-beta',str(beta)])
    if stage in ('all','export'):
        launch('task2_ppo.export_results',common)
        launch('task2_ppo.plot_results',common)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml')
    p.add_argument('--stage',choices=['all','standard','clipping','kl','export'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run_stage(a.stage,a.config,a.resume)
if __name__=='__main__':main()

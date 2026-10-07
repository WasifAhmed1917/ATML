"""Launch matched short beta forks from the original initialization."""
import argparse
from pathlib import Path
from common.data import load_yaml
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


def beta_name(beta): return f'beta_{beta:.2f}'


def run_beta_study(config='configs/dpo.yaml',resume=False):
    cfg=load_yaml(config)
    for beta in cfg['betas']:
        name=beta_name(float(beta))
        output=str(Path(cfg['standard_output']).parent/name)
        run_training(config,name,output_path=output,beta=float(beta),
                     max_examples=int(cfg['short_ablation_examples']),resume=resume,skip_completed=resume)
        run_evaluation(config,output,name,resume=resume)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/dpo.yaml')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run_beta_study(a.config,a.resume)
if __name__=='__main__':main()

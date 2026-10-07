"""Supplied balanced-subset intervention; fixed-stratum and word-limit comparison."""
import argparse
from common.data import load_yaml
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


def run_length_study(config='configs/dpo.yaml',resume=False):
    cfg=load_yaml(config)
    run_training(config,'length_balanced',dataset_path=cfg['paths']['dpo_length_train'],
                 output_path=cfg['length_output'],resume=resume,skip_completed=resume)
    # Standard is evaluated with --length from its first evaluation to keep one protocol.
    run_evaluation(config,cfg['standard_output'],'standard',length=True,resume=resume)
    run_evaluation(config,cfg['length_output'],'length_balanced',length=True,resume=resume)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/dpo.yaml')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run_length_study(a.config,a.resume)
if __name__=='__main__':main()

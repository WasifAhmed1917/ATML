"""Separate GPU subprocesses with live progress and stage-level resume."""
import argparse
from task2_ppo.run_all import launch


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--stage',choices=['gsm','diagnostics','transfer','export','all'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();base=['--config',a.config];flags=['--resume'] if a.resume else []
    if a.stage in ('all','gsm','diagnostics','transfer'):launch('task5_feedback.validate_data',base)
    if a.stage in ('all','gsm'):launch('task5_feedback.evaluate_math',base+['--dataset','gsm',*flags])
    if a.stage in ('all','diagnostics'):launch('task5_feedback.score_perturbations',base+flags)
    if a.stage in ('all','transfer'):launch('task5_feedback.evaluate_math',base+['--dataset','transfer',*flags])
    if a.stage in ('all','export'):
        launch('task5_feedback.compare_feedback',base);launch('task5_feedback.plot_results',base)

if __name__=='__main__':main()

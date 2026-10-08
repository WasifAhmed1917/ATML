"""Run Task 3 stages in separate GPU processes with live Colab progress."""
import argparse
from common.data import load_yaml
from task2_ppo.run_all import launch


def run_stage(stage,config='configs/grpo.yaml',resume=False):
    cfg=load_yaml(config);common=['--config',config];flags=['--resume'] if resume else []
    def train_eval(name,extra):
        adapter=f'outputs/task3_grpo/{name}'
        launch('task3_grpo.continue_train',common+['--run-name',name,'--output',adapter,'--skip-completed',*flags,*extra])
        launch('task3_grpo.evaluate',common+['--adapter',adapter,'--name',name,*flags])
    if stage in ('all','standard'):
        launch('task3_grpo.evaluate',common+['--name','midpoint',*flags]);train_eval('standard',[])
    if stage in ('all','groups'):launch('task3_grpo.analyze_groups',common)
    if stage in ('all','normalization'):
        for loss_type in ('grpo','dr_grpo'):
            train_eval('norm_'+loss_type,['--updates',str(cfg['fork_updates']),'--loss-type',loss_type])
    if stage in ('all','export'):
        launch('task3_grpo.export_results',common);launch('task3_grpo.plot_results',common)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml')
    p.add_argument('--stage',choices=['all','standard','groups','normalization','export'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run_stage(a.stage,a.config,a.resume)
if __name__=='__main__':main()

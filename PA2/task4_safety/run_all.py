"""Sequential GPU processes and separate blind-audit and AI-label stages."""
import argparse
from task2_ppo.run_all import launch
from task4_safety.runtime import POLICIES


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--stage',choices=['generate','audit','judge','export','all'],default='all');p.add_argument('--resume',action='store_true')
    a=p.parse_args();base=['--config',a.config];flags=['--resume'] if a.resume else []
    if a.stage in ('all','generate'):
        launch('task4_safety.validate_data',base)
        for name in POLICIES:launch('task4_safety.generate_responses',base+['--policy',name,*flags])
    if a.stage in ('all','audit'):launch('task4_safety.make_audit_sheet',base)
    if a.stage in ('all','judge'):
        for name in POLICIES:launch('task4_safety.judge_responses',base+['--policy',name,*flags])
    if a.stage in ('all','export'):
        launch('task4_safety.evaluate_safety',base);launch('task4_safety.plot_results',base)
if __name__=='__main__':main()

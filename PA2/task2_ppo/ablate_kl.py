"""Run the released matched eight-update KL forks and held-out evaluations."""
import argparse
from task2_ppo.run_all import run_stage

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run_stage('kl',a.config,a.resume)
if __name__=='__main__':main()

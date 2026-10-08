"""Figures from saved numeric PPO results; no report prose generated."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common.data import load_yaml
from task2_ppo.runtime import artifact_path


def plot(config='configs/ppo.yaml'):
    root=artifact_path(load_yaml(config)['results_dir']);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    history=pd.read_csv(root/'training_trajectories.csv');summary=pd.read_csv(root/'summary.csv')
    metrics=['reward','sampled_kl_token_mean','policy_loss','value_loss','entropy','clip_fraction','policy_gradient_norm','value_gradient_norm','response_length']
    fig,axes=plt.subplots(3,3,figsize=(12,9))
    standard=history[history.condition=='standard']
    for ax,key in zip(axes.flat,metrics):ax.plot(standard['update'],standard[key],marker='.');ax.set(xlabel='Update',ylabel=key);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'standard_trajectories.png',dpi=160);plt.close(fig)
    for prefix,column in [('clip_','epsilon'),('kl_','kl_beta')]:
        rows=summary[summary.condition.str.startswith(prefix)].sort_values(column)
        fig,axes=plt.subplots(1,4,figsize=(13,3))
        for ax,key in zip(axes,['reward_score_mean','sampled_kl_token_mean','entropy_token_mean','response_length_mean']):
            ax.plot(rows[column],rows[key],marker='o');ax.set(xlabel=column,ylabel=key);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/f'{prefix}heldout.png',dpi=160);plt.close(fig)
    cached=pd.read_csv(root/'cached_clipping.csv');fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,key in zip(axes,['affected_token_fraction','active_clip_fraction']):
        ax.plot(cached.epsilon,cached[key],marker='o');ax.set(xlabel='epsilon',ylabel=key)
    fig.tight_layout();fig.savefig(out/'cached_clipping.png',dpi=160);plt.close(fig)
    print(f'Figures saved: {out}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml');a=p.parse_args();plot(a.config)
if __name__=='__main__':main()

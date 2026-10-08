"""Figures from saved Task 3 results; no report narrative generated."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common.data import load_yaml
from task3_grpo.runtime import artifact_path


def plot(config='configs/grpo.yaml'):
    root=artifact_path(load_yaml(config)['results_dir']);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    history=pd.read_csv(root/'training_trajectories.csv');summary=pd.read_csv(root/'summary.csv')
    keys=['reward','sampled_kl_log_ratio','group_reward_std','uninformative_group_fraction','policy_term','gradient_norm','entropy','response_length','truncated_fraction']
    fig,axes=plt.subplots(3,3,figsize=(12,9));standard=history[history.condition=='standard']
    for ax,key in zip(axes.flat,keys):ax.plot(standard['update'],standard[key],marker='.');ax.set(xlabel='Update',ylabel=key);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'standard_trajectories.png',dpi=160);plt.close(fig)
    cached=pd.read_csv(root/'group_sizes'/'summary.csv');fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,key in zip(axes,['informative_group_rate','group_reward_std_mean','group_relative_signal_variance_mean']):
        for name,rows in cached.groupby('difficulty_bin'):ax.plot(rows.k,rows[key],marker='o',label=name)
        ax.set(xlabel='K',ylabel=key,xticks=[2,4,8]);ax.legend(fontsize=7);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'group_sizes.png',dpi=160);plt.close(fig)
    rows=summary[summary.condition.str.startswith('norm_')];fig,axes=plt.subplots(1,4,figsize=(13,3))
    for ax,key in zip(axes,['reward_score_mean','sampled_kl_token_mean','entropy_token_mean','response_length_mean']):
        ax.bar(rows.condition,rows[key]);ax.set(ylabel=key);ax.tick_params(axis='x',labelrotation=20)
    fig.tight_layout();fig.savefig(out/'normalization_heldout.png',dpi=160);plt.close(fig)
    lengths=pd.read_csv(root/'length_gradient_statistics.csv');fig,axes=plt.subplots(1,2,figsize=(10,3))
    for name,rows in lengths[lengths.condition.str.startswith('norm_')].groupby('condition'):
        for ax,key in zip(axes,['policy_logp_abs_gradient_mean','policy_logp_abs_gradient_sum']):
            ax.scatter(rows.response_length,rows[key],s=12,alpha=.5,label=name);ax.set(xlabel='Completion tokens',ylabel=key)
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'normalization_length_gradient.png',dpi=160);plt.close(fig)
    print(f'Figures saved: {out}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args();plot(a.config)
if __name__=='__main__':main()

"""Plot saved numeric results without training, generation, or report interpretation."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common.data import load_yaml
from task1_dpo.utils import artifact_path


def plot_results(config='configs/dpo.yaml'):
    root=artifact_path(load_yaml(config)['results_dir']); dest=root/'figures';dest.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(root/'summary.csv')
    beta=df[df.condition.str.startswith('beta_')].sort_values('beta')
    if len(beta)!=3: raise ValueError('Need all three matched beta runs')
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,key,title in zip(axes.flat,
            ['heldout_dpo_loss','preference_accuracy','sampled_kl_token_mean','reward_score_mean','response_length_mean','generation_truncated_fraction'],
            ['Held-out DPO loss','Preference accuracy','Sampled KL per token','Mean reward score','Mean response tokens','Generation cut-off fraction']):
        ax.plot(beta.beta,beta[key],marker='o',color='black')
        ax.set(xlabel='Beta',ylabel=title,xticks=beta.beta)
        ax.grid(alpha=.2)
    fig.suptitle('Matched 600-pair beta runs')
    fig.savefig(dest/'beta_metrics.png',dpi=180);fig.savefig(dest/'beta_metrics.pdf');plt.close(fig)
    strata=pd.read_csv(root/'length_strata.csv')
    selected=strata[strata.condition.isin(['standard','length_balanced'])]
    table=selected.pivot(index='length_stratum',columns='condition',values='preference_accuracy')
    table=table.reindex(['preferred_longer','length_matched','rejected_longer'])
    ax=table.plot.bar(color=['black','gray'],figsize=(7,4),rot=0)
    ax.set(xlabel='Held-out length stratum',ylabel='Preference accuracy',ylim=(0,1))
    ax.figure.tight_layout();ax.figure.savefig(dest/'length_strata.png',dpi=180);plt.close(ax.figure)
    words=pd.read_csv(root/'word_limit_summary.csv')
    words=words[words.condition.isin(['standard','length_balanced'])].set_index('condition')
    fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained')
    for ax,key,label in zip(axes,['response_length_mean','word_limit_compliance'],['Mean response tokens','Word-limit compliance']):
        ax.bar(words.index,words[key],color=['black','gray']);ax.set_ylabel(label)
        if key=='word_limit_compliance':ax.set_ylim(0,1)
    fig.savefig(dest/'word_limits.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for condition in ['standard','beta_0.03','beta_0.10','beta_0.30','length_balanced']:
        log=pd.read_csv(root/condition/'training_log.csv')
        ax.plot(log.optimizer_step,log.loss,label=condition)
    ax.set(xlabel='Optimizer step',ylabel='Training DPO loss');ax.legend()
    fig.savefig(dest/'training_loss.png',dpi=180);plt.close(fig)
    print(f'Saved numeric plots to {dest}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/dpo.yaml')
    a=p.parse_args();plot_results(a.config)
if __name__=='__main__':main()

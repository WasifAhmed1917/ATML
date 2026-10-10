"""Standalone report figures from validated Task 5 results."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.data import load_yaml
from task5_feedback.runtime import output_dir,POLICIES


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');a=p.parse_args()
    out=output_dir(load_yaml(a.config));result=json.loads((out/'comparison.json').read_text());plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,key,title in zip(axes,['exact_accuracy','format_compliance','response_length_mean'],['Exact final accuracy','Format compliance','Mean response tokens']):
        x=np.arange(3)
        for off,domain,label in [(-.18,'in_domain','GSM8K'),(.18,'transfer','SVAMP')]:
            ax.bar(x+off,[result[domain]['policies'][n][key] for n in POLICIES],width=.36,label=label)
        ax.set_xticks(x,POLICIES);ax.set_title(title)
        if key!='response_length_mean':ax.set_ylim(0,1)
    axes[0].legend();fig.savefig(plots/'policy_comparison.png',dpi=180);fig.savefig(plots/'policy_comparison.pdf');plt.close(fig)
    diag=result['diagnostics']['categories'];fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    variants=['clean_correct','corrupt_reasoning_correct_final','good_reasoning_wrong_final','persuasive_filler_correct','gold_distractor_wrong_final']
    labels=['Clean self-pair','Corrupt reasoning','Wrong final','Persuasive filler','Gold distractor']
    for ax,mechanism in zip(axes,['verifier','judge']):
        bottom=np.zeros(5)
        for key,label,color in [('clean_preference_rate','Clean preferred','#2b8cbe'),('tie_rate','Tie','#aaa'),('variant_preference_rate','Variant preferred','#d95f0e')]:
            values=np.array([diag[mechanism][v][key] for v in variants]);ax.barh(labels,values,left=bottom,label=label,color=color);bottom+=values
        ax.set_xlim(0,1);ax.set_title(mechanism.capitalize());ax.invert_yaxis()
    axes[1].legend(loc='lower left',bbox_to_anchor=(0,-.35),ncols=3)
    fig.savefig(plots/'diagnostic_preferences.png',dpi=180,bbox_inches='tight');fig.savefig(plots/'diagnostic_preferences.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,domain in zip(axes,['in_domain','transfer']):
        x=np.arange(2);bottom=np.zeros(2)
        for key,label in [('ai_win_rate_vs_sft','Win'),('ai_tie_rate_vs_sft','Tie'),('ai_loss_rate_vs_sft','Loss')]:
            values=np.array([result[domain]['policies'][n][key] for n in ('rlvr','rlaif')]);ax.bar(x,values,bottom=bottom,label=label);bottom+=values
        ax.set_xticks(x,['rlvr','rlaif']);ax.set_ylim(0,1);ax.set_title('GSM8K' if domain=='in_domain' else 'SVAMP')
    axes[0].legend();fig.savefig(plots/'judge_vs_sft.png',dpi=180);fig.savefig(plots/'judge_vs_sft.pdf');plt.close(fig)

if __name__=='__main__':main()

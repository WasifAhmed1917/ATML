"""Safety rates retain separate safe/unsafe denominators and all categorical labels."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common.data import load_yaml
from task4_safety.runtime import output_dir,POLICIES,LABEL_ORDER


def plot(config='configs/feedback.yaml'):
    out=output_dir(load_yaml(config));df=pd.read_csv(out/'summary.csv').set_index('policy').loc[list(POLICIES)]
    plots=out/'plots';plots.mkdir(exist_ok=True)
    cols=['safe_answer_rate','safe_over_refusal_rate','unsafe_compliance_rate','unsafe_justified_refusal_rate','ambiguous_judge_rate']
    ax=df[cols].rename(columns={c:c.replace('_',' ') for c in cols}).plot.bar(figsize=(11,5),ylim=(0,1))
    ax.set_ylabel('Rate');ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left');plt.tight_layout();plt.savefig(plots/'calibration_rates.png',dpi=160);plt.close()
    cat=pd.read_csv(out/'category_labels.csv');fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True)
    for ax,policy in zip(axes.flat,POLICIES):
        one=cat[cat.policy==policy].copy();one['category']=one.benchmark_class+' / '+one.type
        pivot=one.pivot(index='category',columns='label',values='rate').reindex(columns=LABEL_ORDER)
        pivot.plot.barh(stacked=True,ax=ax,legend=False);ax.set_title(policy);ax.set_xlim(0,1);ax.set_ylabel('');ax.tick_params(labelsize=8)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=(0,0.05,1,1));fig.savefig(plots/'category_labels.png',dpi=160);plt.close(fig)
    audit=json.loads((out/'manual_audit_summary.json').read_text())
    if audit['status']=='complete':
        cm=pd.read_csv(out/'audit_confusion.csv');cm=cm[cm.policy=='all'].pivot(index='manual_label',columns='ai_label',values='count').reindex(index=LABEL_ORDER,columns=LABEL_ORDER)
        fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(5),LABEL_ORDER,rotation=45,ha='right');ax.set_yticks(range(5),LABEL_ORDER);ax.set_xlabel('AI label');ax.set_ylabel('Manual label')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center')
        fig.tight_layout();fig.savefig(plots/'audit_confusion.png',dpi=160);plt.close(fig)
    print('Safety plots saved:',plots)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');a=p.parse_args();plot(a.config)
if __name__=='__main__':main()

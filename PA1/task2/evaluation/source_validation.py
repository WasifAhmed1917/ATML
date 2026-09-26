# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 12
histories=pd.concat([pd.read_csv(OUT/(name.lower().replace(' ','_').replace('.','p').replace('-','_')+'_history.csv')) for name,_ in RUNS],ignore_index=True)
fig,ax=plt.subplots(1,3,figsize=(15,4))
for name,g in histories.groupby('method',sort=False):
    for a,key in zip(ax,['classification','alignment_or_domain','mean_source_macro_f1']):
        a.plot(g.epoch,g[key],label=name)
for a,title in zip(ax,['Source CE','Alignment / domain CE','Mean source-validation macro-F1']):a.set(xlabel='Epoch',title=title);a.grid(alpha=.25)
ax[-1].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.tight_layout()
fig.savefig(OUT/'training_curves.png',dpi=160,bbox_inches='tight');plt.show()
source_rows=[]
for name,_ in RUNS:
    safe=name.lower().replace(' ','_').replace('.','p').replace('-','_')
    ck=torch.load(OUT/f'{safe}.pt',map_location='cpu',weights_only=True)
    r={'method':name,'best_epoch':ck['epoch']}
    for d in DOMAINS:
        for metric in ('accuracy','macro_f1'):r[f'{d}_{metric}']=ck['source_validation'][d][metric]
    for metric in ('accuracy','macro_f1'):r[f'mean_source_{metric}']=np.mean([r[f'{d}_{metric}'] for d in DOMAINS])
    source_rows.append(r)
source_table=pd.DataFrame(source_rows);source_table.to_csv(OUT/'source_validation.csv',index=False)
display(source_table.round(4))

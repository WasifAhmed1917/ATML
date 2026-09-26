# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 7
def key(name):return name.lower().replace(' ','_').replace('.','p').replace('-','_')
NAMES=('ERM','DAN-DG 0.1','DAN-DG 1','DAN-DG 10','SAM')
def selected(name):
    path=TASK2/'source_only_erm.pt' if name=='ERM' else OUT/f'{key(name)}.pt'
    ck=torch.load(path,map_location=DEVICE,weights_only=True)
    net=Net().to(DEVICE);net.load_state_dict(ck['model']);net.eval()
    return net,ck
source_rows=[]
for name in NAMES:
    net,ck=selected(name)
    scores=source_metrics(net)
    row={'method':name,'selected_epoch':ck['epoch']}
    for d in DOMAINS:
        for metric in ('accuracy','macro_f1'):row[f'{d}_{metric}']=scores[d][metric]
    for metric in ('accuracy','macro_f1'):
        vals=[scores[d][metric] for d in DOMAINS]
        row[f'mean_source_{metric}']=float(np.mean(vals))
        row[f'worst_source_{metric}']=float(np.min(vals))
        row[f'worst_{metric}_domain']=DOMAINS[int(np.argmin(vals))]
    source_rows.append(row)
source_table=pd.DataFrame(source_rows)
source_table.to_csv(OUT/'source_validation.csv',index=False)
display(source_table.round(4))

history_parts=[]
erm_hist=pd.read_csv(TASK2/'source_only_history.csv').copy()
erm_hist['method']='ERM';erm_hist['mmd']=0.;history_parts.append(erm_hist)
for name in NAMES[1:]:history_parts.append(pd.read_csv(OUT/f'{key(name)}_history.csv'))
histories=pd.concat(history_parts,ignore_index=True)
fig,axes=plt.subplots(1,3,figsize=(15,3.7))
for name,frame in histories.groupby('method',sort=False):
    for ax,column in zip(axes,('classification','mmd','mean_source_macro_f1')):
        ax.plot(frame.epoch,frame[column],label=name)
for ax,title in zip(axes,('Source CE','Pairwise source MMD','Mean source-validation macro-F1')):
    ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[-1].legend(bbox_to_anchor=(1.01,1),loc='upper left')
fig.tight_layout();fig.savefig(OUT/'source_training_curves.png',dpi=170,bbox_inches='tight');plt.show()

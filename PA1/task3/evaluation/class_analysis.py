# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 13
for name in ('DAN-DG 1','SAM'):
    changes=per_class[per_class.method==name].sort_values('change_vs_erm_pp')
    negative=changes[changes.change_vs_erm_pp<0].head(2)
    positive=changes[changes.change_vs_erm_pp>0].tail(2)
    print('\n',name,'largest decreases:',list(zip(negative['class'],negative.change_vs_erm_pp.round(1))),
          '| largest increases:',list(zip(positive['class'],positive.change_vs_erm_pp.round(1))))
    for row in pd.concat([negative,positive]).itertuples():
        errors=confusions[(confusions.method==name)&(confusions.actual==row._2)&
                          (confusions.predicted!=row._2)&(confusions.n>0)]
        errors=errors.sort_values('n',ascending=False).head(2)
        print(' ',row._2,[(r.predicted,int(r.n)) for r in errors.itertuples()])

comparisons=pd.DataFrame([
    {'task':'Task 2 DAN (target available unlabeled)','method':'DAN 1',
     'sketch_accuracy':float(original.loc[original.method=='DAN 1','target_accuracy'].iloc[0]),
     'sketch_macro_f1':float(original.loc[original.method=='DAN 1','target_macro_f1'].iloc[0])},
    {'task':'Task 3 DAN-DG (no target access)','method':'DAN-DG 1',
     'sketch_accuracy':float(result.loc[result.method=='DAN-DG 1','sketch_accuracy'].iloc[0]),
     'sketch_macro_f1':float(result.loc[result.method=='DAN-DG 1','sketch_macro_f1'].iloc[0])},
])
comparisons.to_csv(OUT/'task2_dan_vs_task3_dan_dg.csv',index=False)
display(comparisons.round(4))

ref=predictions[predictions.method=='ERM'].set_index('target_file')
examples=[]
for name in ('DAN-DG 1','SAM'):
    curr=predictions[predictions.method==name].set_index('target_file')
    for label,mask in [('recovered',(ref.true!=ref.predicted)&(curr.true==curr.predicted)),
                       ('regressed',(ref.true==ref.predicted)&(curr.true!=curr.predicted))]:
        paths=sorted(ref.index[mask])
        if paths:
            path=paths[0]
            examples.append((name,label,path,curr.loc[path,'true'],
                             ref.loc[path,'predicted'],curr.loc[path,'predicted']))
if examples:
    fig,axes=plt.subplots(1,len(examples),figsize=(3.4*len(examples),3.7),squeeze=False)
    for ax,(name,label,path,true,erm,new) in zip(axes[0],examples):
        with Image.open(ROOT/path) as im:ax.imshow(im.convert('RGB'))
        ax.set_title(f'{name}: {label}\nTrue {true} | ERM {erm} | new {new}',fontsize=9)
        ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'sketch_failure_examples.png',dpi=170,bbox_inches='tight');plt.show()
pd.DataFrame(examples,columns=['method','category','target_file','true',
                              'erm_prediction','method_prediction']).to_csv(OUT/'sketch_failure_examples.csv',index=False)
print('Task 3 checkpoints, splits reference, diagnostics, figures, and tables saved:',OUT)

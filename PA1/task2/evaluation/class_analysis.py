# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 20
study=results[results.method.isin(['DAN 0.1','DAN 1','DAN 10'])][[
 'method','mean_source_accuracy','mean_source_macro_f1','heldout_domain_accuracy',
 'target_accuracy','target_macro_f1','target_accuracy_change_pp']]
study.to_csv(OUT/'dan_strength_study.csv',index=False)
display(study.round(4))
for name in [r[0] for r in RUNS[1:]]:
    changed=per_class[per_class.method==name].sort_values('change_from_source_only_pp')
    selected=pd.concat([changed.head(2),changed.tail(2)])
    print('\n',name)
    for _,row in selected.iterrows():
        errors=confusions[(confusions.method==name)&(confusions.actual==row['class'])&
                          (confusions.actual!=confusions.predicted)].sort_values('n',ascending=False).head(2)
        print(row['class'],f"{row['change_from_source_only_pp']:+.1f} pp;",
              'dominant confusions:',[(r.predicted,int(r.n)) for r in errors.itertuples()])

pr=pd.DataFrame(prediction_rows);reference=pr[pr.method=='Source-only'].set_index('target_file')
examples=[]
for name in ('DAN 1','DANN','CDAN'):
    current=pr[pr.method==name].set_index('target_file')
    for category,mask in [('recovered',(reference.true!=reference.predicted)&(current.true==current.predicted)),
                          ('regressed',(reference.true==reference.predicted)&(current.true!=current.predicted))]:
        paths=sorted(reference.index[mask])
        if paths:examples.append((name,category,paths[0],current.loc[paths[0],'true'],
                                  reference.loc[paths[0],'predicted'],current.loc[paths[0],'predicted']))
if examples:
    fig,axes=plt.subplots(1,len(examples),figsize=(3.1*len(examples),3.4),squeeze=False)
    for ax,(name,category,path,actual,old,new) in zip(axes[0],examples):
        with Image.open(ROOT/path) as im:ax.imshow(im.convert('RGB'))
        ax.set_title(f'{name}: {category}\nTrue {actual} | ERM {old} | {name} {new}',fontsize=8);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'target_failure_examples.png',dpi=170,bbox_inches='tight');plt.show()
pd.DataFrame(examples,columns=['method','category','target_file','true','source_only_prediction','method_prediction']).to_csv(OUT/'target_failure_examples.csv',index=False)
print('Saved all Task 2 tables, curves, checkpoints, splits and selected examples:',OUT)

# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 11
assert all((OUT/f'{key(name)}_done.json').is_file() and (OUT/f'{key(name)}.pt').is_file()
           for name in NAMES[1:]),'Complete all Task 3 source-only training before final evaluation.'
assert (OUT/'source_validation.csv').is_file() and (OUT/'source_diagnostics.csv').is_file()
target_dir=ROOT/'sketch'
assert target_dir.is_dir(),'Expected Sketch only at the final evaluation stage.'
target_paths=sorted(str(p.relative_to(ROOT)) for p in target_dir.rglob('*')
                    if p.is_file() and p.suffix.lower() in extensions)
assert len(target_paths)==t2config['dataset_domain_counts']['sketch']==3929
target=Images(target_paths,eval_tf)
result_rows=[];class_rows=[];cm_rows=[];prediction_rows=[]
for name in NAMES:
    net,_=selected(name)
    true,pred=predict(net,target)
    assert len(true)==len(target_paths) and set(np.unique(true))==set(range(7))
    result_rows.append({'method':name,'sketch_accuracy':accuracy_score(true,pred),
                        'sketch_macro_f1':f1_score(true,pred,labels=range(7),average='macro',zero_division=0)})
    matrix=confusion_matrix(true,pred,labels=range(7))
    for i,actual in enumerate(CLASSES):
        sel=true==i
        class_rows.append({'method':name,'class':actual,'n':int(sel.sum()),
                           'accuracy':float(np.mean(pred[sel]==i))})
        for j,predicted in enumerate(CLASSES):
            cm_rows.append({'method':name,'actual':actual,'predicted':predicted,
                            'n':int(matrix[i,j])})
    prediction_rows.extend({'method':name,'target_file':path,'true':CLASSES[int(y)],
                            'predicted':CLASSES[int(p)]}
                           for path,y,p in zip(target_paths,true,pred))
result_frame=pd.DataFrame(result_rows)
result=source_table.merge(diagnostics.drop(columns=['validation_examples_per_domain']),on='method').merge(result_frame,on='method')
erm_acc=result.loc[result.method=='ERM','sketch_accuracy'].iloc[0]
result['sketch_change_vs_erm_pp']=100*(result.sketch_accuracy-erm_acc)
original=pd.read_csv(TASK2/'task2_summary.csv')
t2_erm=original.loc[original.method=='Source-only','target_accuracy'].iloc[0]
assert abs(erm_acc-t2_erm)<1e-12,'Reused ERM differs from Task 2 target evaluation.'
result.to_csv(OUT/'task3_summary.csv',index=False)
per_class=pd.DataFrame(class_rows)
erm_class=per_class[per_class.method=='ERM'].set_index('class')['accuracy']
per_class['change_vs_erm_pp']=per_class.apply(lambda r:100*(r.accuracy-erm_class[r['class']]),axis=1)
per_class.to_csv(OUT/'sketch_per_class.csv',index=False)
confusions=pd.DataFrame(cm_rows);confusions.to_csv(OUT/'sketch_confusions.csv',index=False)
predictions=pd.DataFrame(prediction_rows);predictions.to_csv(OUT/'sketch_predictions.csv',index=False)
print('Main method comparison:')
display(result[result.method.isin(['ERM','DAN-DG 1','SAM'])].round(4))
print('Locked DAN-DG strength study (main λ=1):')
study=result[result.method.str.startswith('DAN-DG')][['method','mean_source_macro_f1',
    'worst_source_macro_f1','source_domain_separability','sharpness_increase',
    'sketch_accuracy','sketch_macro_f1','sketch_change_vs_erm_pp']]
study.to_csv(OUT/'dan_dg_strength_study.csv',index=False)
display(study.round(4))
print('Sketch class-level changes (percentage points):')
display(per_class.pivot(index='class',columns='method',values='change_vs_erm_pp').round(1))

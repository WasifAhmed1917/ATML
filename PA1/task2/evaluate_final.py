# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 18
assert lock_file.is_file(),'Lock Task 3 settings above first.'
assert all((OUT/(name.lower().replace(' ','_').replace('.','p').replace('-','_')+'_done.json')).is_file() for name,_ in RUNS)
target_eval=Images(target_rel,eval_tf,labeled=True)
rows=[];class_rows=[];confusion_rows=[];prediction_rows=[]
for name,_ in RUNS:
    net=load_selected(name)
    yt=[];yp=[]
    for x,y in loader(target_eval,64):
        with torch.inference_mode(): _,logits=net(x.to(DEVICE,non_blocking=True))
        yt.extend(y.numpy());yp.extend(logits.argmax(1).cpu().numpy())
    yt=np.asarray(yt);yp=np.asarray(yp)
    assert len(yt)==len(target_rel) and set(np.unique(yt))==set(range(7))
    rows.append({'method':name,'target_accuracy':accuracy_score(yt,yp),
                 'target_macro_f1':f1_score(yt,yp,labels=range(7),average='macro',zero_division=0)})
    for i,cl in enumerate(CLASSES):
        sel=yt==i
        class_rows.append({'method':name,'class':cl,'n':int(sel.sum()),'accuracy':float((yp[sel]==i).mean())})
    cm=confusion_matrix(yt,yp,labels=range(7))
    for i,actual in enumerate(CLASSES):
        for j,predicted in enumerate(CLASSES):
            confusion_rows.append({'method':name,'actual':actual,'predicted':predicted,'n':int(cm[i,j])})
    prediction_rows.extend({'method':name,'target_file':rel,'true':CLASSES[int(y)],'predicted':CLASSES[int(p)]}
                           for rel,y,p in zip(target_rel,yt,yp))
    del net
results=source_table.merge(pd.DataFrame(rows),on='method').merge(separability[['method','heldout_domain_accuracy']],on='method')
baseline=float(results.loc[results.method=='Source-only','target_accuracy'].iloc[0])
results['target_accuracy_change_pp']=100*(results.target_accuracy-baseline)
results.to_csv(OUT/'task2_summary.csv',index=False)
per_class=pd.DataFrame(class_rows);base=per_class[per_class.method=='Source-only'].set_index('class')['accuracy']
per_class['change_from_source_only_pp']=per_class.apply(lambda r:100*(r.accuracy-base[r['class']]),axis=1)
per_class.to_csv(OUT/'target_per_class.csv',index=False)
confusions=pd.DataFrame(confusion_rows);confusions.to_csv(OUT/'target_confusions.csv',index=False)
pd.DataFrame(prediction_rows).to_csv(OUT/'target_predictions.csv',index=False)
display(results.round(4))
display(per_class.pivot(index='class',columns='method',values='change_from_source_only_pp').round(1))
print('Largest per-class changes for each adapted method:')
for name,_ in RUNS[1:]:
    g=per_class[per_class.method==name].sort_values('change_from_source_only_pp')
    print(name,'decreases:',list(zip(g['class'].head(2),g.change_from_source_only_pp.head(2).round(1))),
          'increases:',list(zip(g['class'].tail(2),g.change_from_source_only_pp.tail(2).round(1))))

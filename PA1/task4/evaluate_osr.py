# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 11
c100=torchvision.datasets.CIFAR100('/content/data',train=False,download=True)
assert len(c100)==10000 and len(c100.classes)==100
near_names=['bus','pickup_truck','motorcycle','tractor','wolf','fox','leopard','camel']
far_names=['bottle','bowl','chair','clock','keyboard','mushroom','sunflower','wardrobe']
near_classes={c100.class_to_idx[n] for n in near_names}
far_classes={c100.class_to_idx[n] for n in far_names}
near_ids=np.flatnonzero(np.isin(c100.targets,list(near_classes)))
far_ids=np.flatnonzero(np.isin(c100.targets,list(far_classes)))
assert len(near_ids)==len(far_ids)==800 and not set(near_ids)&set(far_ids)
for group,group_ids in [('near',near_ids),('far',far_ids)]:
    counts=pd.Series(np.asarray(c100.targets)[group_ids]).value_counts()
    assert len(counts)==8 and counts.eq(100).all()
np.savez(RESULTS/'cifar100_test_ids.npz',near_ids=near_ids,far_ids=far_ids)
near_ds=Indexed(c100,near_ids,clean_transform); far_ds=Indexed(c100,far_ids,clean_transform)
vnear=extract(vanilla,near_ds,ARRAYS/'vanilla_near.npz',features=True)
vfar=extract(vanilla,far_ds,ARRAYS/'vanilla_far.npz',features=True)
gnear=extract(gcsc,near_ds,ARRAYS/'gcsc_near.npz')
gfar=extract(gcsc,far_ds,ARRAYS/'gcsc_far.npz')
pnear=extract(proser,near_ds,ARRAYS/'proser_near.npz')
pfar=extract(proser,far_ds,ARRAYS/'proser_far.npz')
vscore_near=scores(vnear['logits'],vnear['features'],mu,var)
vscore_far=scores(vfar['logits'],vfar['features'],mu,var)
gscore_near=scores(gnear['logits'])['MLS']; gscore_far=scores(gfar['logits'])['MLS']
pscore_near=scores(pnear['logits'])['MLS']; pscore_far=scores(pfar['logits'])['MLS']
ppscore_near=placeholder_score(pnear['logits']); ppscore_far=placeholder_score(pfar['logits'])

def row(model,score,known,near,far,known_logits,known_y):
    known=np.asarray(known); near=np.asarray(near); far=np.asarray(far)
    tau=thresholds[model+' '+score]
    all_unknown=np.concatenate([near,far])
    def auc(u): return roc_auc_score(np.r_[np.zeros(len(known)),np.ones(len(u))],np.r_[known,u])
    assert all(np.isfinite(a).all() for a in [known,near,far])
    return {'Model':model,'Score':score,'CSA':float((known_logits[:,:10].argmax(1)==known_y).mean()),
            'AUROC near':auc(near),'AUROC far':auc(far),'AUROC all':auc(all_unknown),
            'Known accept':float((known<=tau).mean()),'Near reject':float((near>tau).mean()),
            'Far reject':float((far>tau).mean()),'FPR near @95TPR':float((near<=tau).mean()),
            'FPR far @95TPR':float((far<=tau).mean()),'Threshold':tau}

vanilla_rows=[row('Vanilla',s,vscore_test[s],vscore_near[s],vscore_far[s],vtest['logits'],vtest['labels']) for s in vscore_test]
model_rows=[vanilla_rows[1],row('GCSC','MLS',gscore_test,gscore_near,gscore_far,gtest['logits'],gtest['labels']),
            row('PROSER','MLS',pscore_test,pscore_near,pscore_far,ptest['logits'],ptest['labels']),
            row('PROSER','placeholder',ppscore_test,ppscore_near,ppscore_far,ptest['logits'],ptest['labels'])]
base_table=pd.DataFrame(vanilla_rows).set_index('Score').drop(columns='Model')
model_table=pd.DataFrame(model_rows).set_index(['Model','Score'])
base_table.to_csv(RESULTS/'vanilla_scores.csv'); model_table.to_csv(RESULTS/'trained_models.csv')
print('Vanilla: fixed model, four scores | larger AUROC and rejection are better')
display(base_table.round(4))
print('Trained-model comparison | CSA uses only ten known logits')
display(model_table.round(4))
print('Validation accept rates:',{k:round(float((val<=thresholds[k]).mean()),4) for k,val in {
    **{'Vanilla '+s:v for s,v in vscore_val.items()},'GCSC MLS':gscore_val,
    'PROSER MLS':pscore_val,'PROSER placeholder':ppscore_val}.items()})

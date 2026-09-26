# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 25

COLOR_FEATURES={}; CUE_FEATURES={}; PATCH_FEATURES={}; TRANS16={}; TRANS_CURVES=[]
def logits_for(m,feat):
    if m=='CLIP zero-shot': return scale*(feat@txt.T)
    w,b=HEADS[m]; return feat@w.T+b

def append_result(m,condition,feat,labels,reference):
    result=summarize(logits_for(m,feat),labels,reference)
    baseline=summarize(CLEAN_LOGITS[m],y)['accuracy']
    ROWS.append(dict(model=m,condition=condition,accuracy_change_pp=100*(result['accuracy']-baseline),**result))

CUE_DECISIONS=[]
for backbone in ['ResNet-50','ViT-B/16','CLIP head']:
    print('Evaluating',backbone)
    model=make_model(backbone)
    fgray=features(model,backbone,GRAY); fhue=features(model,backbone,HUE)
    fcue=features(model,backbone,CUE)
    COLOR_FEATURES[backbone]={'grayscale':fgray,'hue +90°':fhue}
    CUE_FEATURES[backbone]=fcue
    translated_16={}
    for delta in [8,16,32]:
        for direction in DIRECTIONS:
            ims=np.stack([translate(im,delta,direction) for im in X])
            fd=features(model,backbone,ims)
            if delta==16: translated_16[direction]=fd
            for m in ([backbone,'CLIP zero-shot'] if backbone=='CLIP head' else [backbone]):
                z=summarize(logits_for(m,fd),y,PRED_CLEAN[m])
                TRANS_CURVES.append(dict(model=m,pixels=delta,direction=direction,accuracy=z['accuracy'],consistency=z['consistency']))
            del ims,fd
    TRANS16[backbone]=translated_16
    fpatch=features(model,backbone,PATCH)
    PATCH_FEATURES[backbone]=fpatch
    for m in ([backbone,'CLIP zero-shot'] if backbone=='CLIP head' else [backbone]):
        append_result(m,'grayscale',fgray,y,PRED_CLEAN[m])
        append_result(m,'hue +90°',fhue,y,PRED_CLEAN[m])
        append_result(m,'patch shuffle',fpatch,y,PRED_CLEAN[m])
        preds=probs(logits_for(m,fcue)).argmax(1)
        for ix,pred in enumerate(preds):
            CUE_DECISIONS.append(dict(model=m,candidate_id=int(ACCEPTED_CANDIDATE[ix]),
                content_id=int(CUE_CONTENT[ix]),style_id=int(accepted.style_id.iloc[ix]),
                content_label=CLASSES[CUE_SHAPE[ix]],texture_label=CLASSES[CUE_TEXTURE[ix]],
                prediction=CLASSES[pred],decision='shape' if pred==CUE_SHAPE[ix] else 'texture' if pred==CUE_TEXTURE[ix] else 'other'))
    del model
    gc.collect(); torch.cuda.empty_cache()
print('Evaluation complete.')

# %% Original notebook cell 27
performance=pd.DataFrame(ROWS)
performance['accuracy_change_pp']=performance['accuracy_change_pp'].fillna(0)
performance.to_csv(OUT/'performance.csv',index=False)
show_table(performance[['model','condition','accuracy','accuracy_change_pp','macro_f1','mean_max_confidence','consistency']].round(4))
print('Accuracy change is in percentage points; consistency compares predictions with the same clean image.')

# %% Original notebook cell 29
cue_df=pd.DataFrame(CUE_DECISIONS); cue_df.to_csv(OUT/'cue_predictions.csv',index=False)
shape_rows=[]
for m,g in cue_df.groupby('model',sort=False):
    z=g.decision.value_counts(); ns=int(z.get('shape',0)); nt=int(z.get('texture',0)); no=int(z.get('other',0))
    shape_rows.append(dict(model=m,shape=ns,texture=nt,other=no,total=len(g),
        shape_bias_pct=100*ns/(ns+nt) if ns+nt else np.nan,coverage_pct=100*(ns+nt)/len(g)))
shape_table=pd.DataFrame(shape_rows);shape_table.to_csv(OUT/'shape_bias.csv',index=False)
show_table(shape_table.round(2))
assert (shape_table[['shape','texture','other']].sum(axis=1)==shape_table.total).all()
show_table(cue_df.groupby(['model','content_label','texture_label','decision']).size().rename('n').unstack(fill_value=0))

piv=cue_df.pivot(index='candidate_id',columns='model',values='decision')
examples=[]
for title,condition in [('shape agreement',(piv=='shape').all(axis=1)),
                         ('mixed decisions',piv.nunique(axis=1)>1),('other-label failure',(piv=='other').any(axis=1))]:
    hits=piv.index[condition]
    if len(hits): examples.append((title,int(hits[0])))
if not examples:
    examples=[('first reviewed conflict',int(piv.index[0]))]
fig,axes=plt.subplots(len(examples),4,figsize=(11,3*len(examples)),squeeze=False)
for row,(title,cid) in enumerate(examples):
    r=review.iloc[cid]
    for col,(im,label) in enumerate([(X[int(r.content_id)],'Original: '+r.content_class),
                                      ((X[int(r.content_id)].astype(np.float32)*(MASKS[cid,:,:,None]/255)+255*(1-MASKS[cid,:,:,None]/255)).astype(np.uint8),'Isolated'),
                                      (texture_tile(X[int(r.style_id)]),'Style: '+r.style_class),(CAND[cid],'Conflict')]):
        axes[row,col].imshow(im); axes[row,col].set_title(label); axes[row,col].axis('off')
    predictions=cue_df[cue_df.candidate_id==cid].set_index('model')['prediction'].to_dict()
    axes[row,3].set_xlabel(f'{title} · id {cid} | {predictions}',fontsize=8)
plt.tight_layout();plt.savefig(OUT/'cue_examples.png',dpi=160,bbox_inches='tight');plt.show()

# %% Original notebook cell 31
curves=pd.DataFrame(TRANS_CURVES)
for m in NAMES:
    baseline=summarize(CLEAN_LOGITS[m],y)
    curves.loc[len(curves)]=dict(model=m,pixels=0,direction='all',accuracy=baseline['accuracy'],consistency=1.0)
curves.to_csv(OUT/'translation_by_direction.csv',index=False)
mean_curve=curves.groupby(['model','pixels'],as_index=False)[['accuracy','consistency']].mean()
mean_curve.to_csv(OUT/'translation_mean.csv',index=False)
show_table(mean_curve.round(4))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for m,g in mean_curve.groupby('model'):
    g=g.sort_values('pixels')
    for a,metric in zip(axes,['accuracy','consistency']): a.plot(g.pixels,100*g[metric],marker='o',label=m)
for a,title in zip(axes,['Top-1 accuracy','Prediction consistency']):
    a.set(xlabel='Displacement (pixels)',ylabel=title+' (%)',xticks=[0,8,16,32]);a.grid(alpha=.3)
axes[1].legend(fontsize=8);plt.tight_layout();plt.savefig(OUT/'translation.png',dpi=160);plt.show()

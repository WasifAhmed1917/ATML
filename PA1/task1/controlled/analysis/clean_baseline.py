# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 10
def probs(logits):
    z=torch.from_numpy(np.asarray(logits,dtype=np.float32))
    return torch.softmax(z,dim=1).numpy()
def summarize(logits,labels,reference=None):
    p=probs(logits); pred=p.argmax(axis=1)
    return dict(accuracy=accuracy_score(labels,pred),macro_f1=f1_score(labels,pred,labels=np.arange(10),average='macro',zero_division=0),
                mean_max_confidence=p.max(axis=1).mean(), consistency=np.nan if reference is None else np.mean(pred==reference))
PRED_CLEAN={m:probs(CLEAN_LOGITS[m]).argmax(1) for m in NAMES}
ROWS=[]
for m in NAMES:
    ROWS.append(dict(model=m,condition='clean',**summarize(CLEAN_LOGITS[m],y)))
def show_table(frame,columns=None):
    display(frame[columns] if columns else frame)
show_table(pd.DataFrame(ROWS).round(4))

# %% Original notebook cell 11
per_class=[]
for name in NAMES:
    for ci,label in enumerate(CLASSES):
        keep=(y==ci)
        per_class.append(dict(model=name,class_name=label,n=int(keep.sum()),
                              accuracy=float((PRED_CLEAN[name][keep]==ci).mean())))
per_class=pd.DataFrame(per_class)
per_class.to_csv(OUT/'clean_per_class.csv',index=False)
display(per_class.pivot(index='class_name',columns='model',values='accuracy').round(3))

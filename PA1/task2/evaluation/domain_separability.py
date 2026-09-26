# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 14
def load_selected(name):
    safe=name.lower().replace(' ','_').replace('.','p').replace('-','_')
    net=Net().to(DEVICE)
    net.load_state_dict(torch.load(OUT/f'{safe}.pt',map_location=DEVICE,weights_only=True)['model'])
    net.eval();return net
@torch.inference_mode()
def collect(net,dataset,labeled):
    feats=[];pred=[];truth=[]
    for batch in loader(dataset,64):
        if labeled:x,y=batch;truth.extend(y.numpy())
        else:x=batch
        f,z=net(x.to(DEVICE,non_blocking=True))
        feats.append(f.cpu().numpy());pred.extend(z.argmax(1).cpu().numpy())
    return np.concatenate(feats),np.asarray(pred),np.asarray(truth) if labeled else None
source_diag_paths=[path for d in DOMAINS for path in splits[d]['val']]
source_diag=Images(source_diag_paths,eval_tf)
sep_rows=[]
for name,_ in RUNS:
    net=load_selected(name)
    fs,_,_=collect(net,source_diag,True)
    ft,_,_=collect(net,target_diagnostic,False)
    n=min(len(fs),len(ft));rng=np.random.default_rng(SEED)
    ix=rng.choice(len(fs),n,replace=False);iy=rng.choice(len(ft),n,replace=False)
    Xdomain=np.concatenate([fs[ix],ft[iy]])
    Ydomain=np.concatenate([np.zeros(n,dtype=int),np.ones(n,dtype=int)])
    train_ix,test_ix=train_test_split(np.arange(2*n),test_size=0.3,stratify=Ydomain,random_state=SEED)
    probe=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=SEED)
    probe.fit(Xdomain[train_ix],Ydomain[train_ix])
    acc=accuracy_score(Ydomain[test_ix],probe.predict(Xdomain[test_ix]))
    sep_rows.append({'method':name,'source_n':n,'target_n':n,'heldout_domain_accuracy':acc})
    del net
separability=pd.DataFrame(sep_rows);separability.to_csv(OUT/'domain_separability.csv',index=False)
display(separability.round(4))

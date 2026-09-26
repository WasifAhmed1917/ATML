# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 8
HEADS={}; CLEAN_FEATURES={}; CLEAN_LOGITS={}; HISTORY=[]
for name in ['ResNet-50','ViT-B/16','CLIP head']:
    print('\n',name)
    model=make_model(name)
    ftr=features(model,name,Xtr); fva=features(model,name,Xva); ftest=features(model,name,X)
    assert len(ftr)==len(tr_idx) and len(fva)==len(va_idx) and len(ftest)==len(X)
    assert np.isfinite(ftr).all() and np.isfinite(ftest).all()

    torch.manual_seed(SEED); np.random.seed(SEED)
    head=nn.Linear(ftr.shape[1],10).to(DEVICE)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    a=torch.from_numpy(ftr).to(DEVICE); b=torch.from_numpy(fva).to(DEVICE)
    ya=torch.from_numpy(y_train[tr_idx]).to(DEVICE); yb=torch.from_numpy(y_train[va_idx]).to(DEVICE)
    best=-1.0; stale=0; best_state=None
    for epoch in range(1,51):
        head.train(); gen=torch.Generator().manual_seed(SEED+epoch)
        perm=torch.randperm(len(a),generator=gen).to(DEVICE)
        for ix in perm.split(128):
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(head(a[ix]),ya[ix]); loss.backward(); opt.step()
        head.eval()
        with torch.inference_mode(): score=(head(b).argmax(1)==yb).float().mean().item()
        HISTORY.append(dict(model=name,epoch=epoch,validation_accuracy=score))
        if score>best+1e-12:
            best=score; stale=0; best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_epoch=epoch
        else: stale+=1
        if stale>=5: break
    head.load_state_dict(best_state); head.eval()
    torch.save({'state_dict':best_state,'feature_dim':ftr.shape[1],'best_epoch':best_epoch,
                'val_accuracy':best,'seed':SEED},OUT/(name.replace('/','_').replace(' ','_')+'_head.pt'))
    HEADS[name]=(head.weight.detach().cpu().numpy(),head.bias.detach().cpu().numpy())
    CLEAN_FEATURES[name]=ftest
    CLEAN_LOGITS[name]=ftest@HEADS[name][0].T+HEADS[name][1]
    print(f'best epoch {best_epoch}, validation accuracy {best:.4f}, feature dim {ftr.shape[1]}')
    if name=='CLIP head':
        tokens=open_clip.get_tokenizer('ViT-B-32')([f'a photo of a {c}.' for c in CLASSES]).to(DEVICE)
        with torch.inference_mode():
            txt=F.normalize(model.net.encode_text(tokens).float(),dim=-1).cpu().numpy()
            scale=model.net.logit_scale.exp().detach().float().cpu().item()
        CLEAN_LOGITS['CLIP zero-shot']=scale*(ftest@txt.T)
        print('CLIP logit scale:',scale)
    del model, ftr, fva, ftest, a, b, ya, yb, head, opt
    gc.collect(); torch.cuda.empty_cache()
pd.DataFrame(HISTORY).to_csv(OUT/'head_training.csv',index=False)
assert set(CLEAN_LOGITS)==set(NAMES)
print('All backbones frozen; heads saved.')

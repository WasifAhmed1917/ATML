# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 9
@torch.inference_mode()
def extract(net,ds,path,features=False):
    if path.exists():
        saved=np.load(path)
        assert len(saved['ids'])==len(ds)
        return {key:saved[key] for key in saved.files}
    net.eval(); ids=[]; labels=[]; logits=[]; feats=[]
    for x,y,i in loader(ds):
        f=net.features(x.to(DEVICE,non_blocking=True))
        z=net.net.fc(f)
        logits.append(z.cpu().numpy()); labels.append(y.numpy()); ids.append(i.numpy())
        if features: feats.append(f.cpu().numpy())
    out={'ids':np.concatenate(ids),'labels':np.concatenate(labels),'logits':np.concatenate(logits)}
    if features: out['features']=np.concatenate(feats)
    assert np.array_equal(out['ids'],ds.ids) and np.isfinite(out['logits']).all()
    np.savez_compressed(path,**out)
    return out

vtrain=extract(vanilla,known_train_clean,ARRAYS/'vanilla_train.npz',features=True)
vval=extract(vanilla,known_val,ARRAYS/'vanilla_val.npz',features=True)
vtest=extract(vanilla,known_test_clean,ARRAYS/'vanilla_test.npz',features=True)
gval=extract(gcsc,known_val,ARRAYS/'gcsc_val.npz')
gtest=extract(gcsc,known_test_clean,ARRAYS/'gcsc_test.npz')
pval=extract(proser,known_val,ARRAYS/'proser_val.npz')
ptest=extract(proser,known_test_clean,ARRAYS/'proser_test.npz')
assert all(len(a['labels'])==5000 for a in (vval,gval,pval))
assert all(len(a['labels'])==10000 for a in (vtest,gtest,ptest))
assert vtrain['features'].shape==(45000,512) and pval['logits'].shape==(5000,15)
print('CIFAR-10 test CSA:',{name:round(float((a['logits'][:,:10].argmax(1)==a['labels']).mean()),4)
                              for name,a in [('Vanilla',vtest),('GCSC',gtest),('PROSER',ptest)]})

mu=np.stack([vtrain['features'][vtrain['labels']==k].mean(0) for k in range(10)])
res=vtrain['features']-mu[vtrain['labels']]
var=np.mean(res.astype(np.float64)**2,axis=0)+1e-6
np.savez(RESULTS/'mahalanobis_train_statistics.npz',mu=mu,variance=var)


def scores(logits,features=None,mu=None,var=None):
    z=np.asarray(logits,dtype=np.float64)[:,:10]
    shift=z-z.max(axis=1,keepdims=True)
    probs=np.exp(shift); probs/=probs.sum(axis=1,keepdims=True)
    result={'MSP':1-probs.max(1),'MLS':-z.max(1),'Energy':-(z.max(1)+np.log(np.exp(shift).sum(1)))}
    if features is not None:
        assert mu is not None and var is not None
        f=np.asarray(features,dtype=np.float64)
        result['Mahalanobis']=np.min(np.sum(((f[:,None,:]-mu[None,:,:])**2)/var[None,None,:],axis=2),axis=1)
    return result

def placeholder_score(logits):
    z=np.asarray(logits,dtype=np.float64)
    assert z.shape[1]==15
    reduced=np.column_stack([z[:,:10],z[:,10:].max(1)])
    shifted=reduced/1024.0
    shifted-=shifted.max(1,keepdims=True)
    p=np.exp(shifted); p/=p.sum(1,keepdims=True)
    return p[:,10]-p[:,:10].max(1)

vscore_val=scores(vval['logits'],vval['features'],mu,var)
vscore_test=scores(vtest['logits'],vtest['features'],mu,var)
gscore_val=scores(gval['logits'])['MLS']; gscore_test=scores(gtest['logits'])['MLS']
pscore_val=scores(pval['logits'])['MLS']; pscore_test=scores(ptest['logits'])['MLS']
ppscore_val=placeholder_score(pval['logits']); ppscore_test=placeholder_score(ptest['logits'])
thresholds={'Vanilla '+key:float(np.percentile(val,95)) for key,val in vscore_val.items()}
thresholds.update({'GCSC MLS':float(np.percentile(gscore_val,95)),
                   'PROSER MLS':float(np.percentile(pscore_val,95)),
                   'PROSER placeholder':float(np.percentile(ppscore_val,95))})
(RESULTS/'frozen_protocol.json').write_text(json.dumps({
    'seed':SEED,'train_ids_file':'cifar10_split_seed6304.npz','normalization_mean':mean.tolist(),
    'normalization_std':std.tolist(),'selected_epoch':{n:int(ck['epoch']) for n,ck in [('Vanilla',vck),('GCSC',gck),('PROSER',pck)]},
    'score_definitions':{'MSP':'1-max softmax(known logits)','MLS':'-max known logit',
       'Energy':'-logsumexp(known logits)','Mahalanobis':'minimum shared diagonal class distance',
       'PROSER placeholder':'softmax([known logits, max dummy]/1024) dummy minus max known; zero bias'},
    'accept_rule':'unknownness <= 95th percentile of CIFAR10 validation unknownness',
    'thresholds':thresholds,'near_classes':['bus','pickup_truck','motorcycle','tractor','wolf','fox','leopard','camel'],
    'far_classes':['bottle','bowl','chair','clock','keyboard','mushroom','sunflower','wardrobe']},indent=2))
print('Known validation thresholds frozen before CIFAR-100 is opened:')
display(pd.Series(thresholds,name='threshold').to_frame().round(6))

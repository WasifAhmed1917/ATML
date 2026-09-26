# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 5
def mmd_rbf(a,b):
    combined=torch.cat([a,b],0)
    sq=torch.cdist(combined,combined,p=2).square()
    with torch.no_grad():
        upper=sq.triu(diagonal=1)
        values=upper[upper>0]
        median=values.median().clamp_min(1e-8) if len(values) else sq.new_tensor(1.)
    kernel=sum(torch.exp(-sq/(2*scale*median)) for scale in (0.5,1.,2.))
    n=len(a)
    return kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean()

def source_batch(its):
    pairs=[next(its[d]) for d in DOMAINS]
    xs,ys=zip(*pairs)
    return torch.cat(xs).to(DEVICE,non_blocking=True),torch.cat(ys).to(DEVICE,non_blocking=True)

@torch.inference_mode()
def predict(net,ds):
    net.eval();true=[];pred=[]
    for x,y in loader(ds,64):
        _,z=net(x.to(DEVICE,non_blocking=True))
        true.extend(y.numpy());pred.extend(z.argmax(1).cpu().numpy())
    return np.asarray(true),np.asarray(pred)

def source_metrics(net):
    results={}
    for d in DOMAINS:
        y,p=predict(net,source_val[d])
        results[d]={'accuracy':accuracy_score(y,p),
                    'macro_f1':f1_score(y,p,labels=range(7),average='macro',zero_division=0)}
    return results

def sam_update(net,opt,x,y,params,rho=0.05):
    opt.zero_grad(set_to_none=True)
    f,z=net(x);first=F.cross_entropy(z,y)
    first.backward()
    with torch.no_grad():
        gnorm=torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]))
        assert bool(torch.isfinite(gnorm))
        scale=rho/gnorm.clamp_min(1e-12)
        perturb={p:scale*p.grad.detach().clone() for p in params if p.grad is not None}
        for p,eps in perturb.items():p.add_(eps)
    try:
        opt.zero_grad(set_to_none=True)
        _,z_pert=net(x);second=F.cross_entropy(z_pert,y)
        assert bool(torch.isfinite(second))
        second.backward()
    finally:
        with torch.no_grad():
            for p,eps in perturb.items():p.sub_(eps)
    grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
    opt.step()
    return float(first.detach()),float(grad),float(f.detach().norm(dim=1).median())

def train(name,strength=0.):
    safe=name.lower().replace(' ','_').replace('.','p').replace('-','_')
    checkpoint=OUT/f'{safe}.pt';history_file=OUT/f'{safe}_history.csv'
    done=OUT/f'{safe}_done.json'
    if done.is_file() and checkpoint.is_file() and history_file.is_file():
        print(name,'completed; using saved source-selected checkpoint')
        return
    net=initial_net();params=list(net.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    steps=max(math.ceil(len(source_train[d])/8) for d in DOMAINS)
    bn=[(m,m.running_mean.clone(),m.running_var.clone()) for m in net.modules()
        if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    best=-float('inf');stale=0;rows=[]
    for epoch in range(1,31):
        net.train();freeze_bn_stats(net)
        src=epoch_loaders(epoch);its={d:forever(src[d]) for d in DOMAINS}
        totals={'classification':0.,'mmd':0.,'median_feature_norm':0.,
                'max_gradient_before_clip':0.}
        for _ in tqdm(range(steps),desc=f'{name} epoch {epoch}',leave=False):
            x,y=source_batch(its)
            if name=='SAM':
                cls,grad,feature_norm=sam_update(net,opt,x,y,params,rho=0.05)
                mmd=0.
            else:
                f,z=net(x);loss_cls=F.cross_entropy(z,y)
                a,b,c=f.split(8,dim=0)
                loss_mmd=(mmd_rbf(a,b)+mmd_rbf(a,c)+mmd_rbf(b,c))/3
                loss=loss_cls+strength*loss_mmd
                assert bool(torch.isfinite(loss)),f'Nonfinite loss: {name} epoch {epoch}'
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
                opt.step()
                cls=float(loss_cls.detach());mmd=float(loss_mmd.detach())
                feature_norm=float(f.detach().norm(dim=1).median())
            assert math.isfinite(feature_norm) and feature_norm<1e5, f'Unstable {name} feature norm.'
            totals['classification']+=cls/steps;totals['mmd']+=mmd/steps
            totals['median_feature_norm']+=feature_norm/steps
            totals['max_gradient_before_clip']=max(totals['max_gradient_before_clip'],grad)
        assert all(torch.equal(m.running_mean,mu) and torch.equal(m.running_var,var)
                   for m,mu,var in bn),'BatchNorm running statistics changed.'
        scores=source_metrics(net)
        mean_f1=float(np.mean([scores[d]['macro_f1'] for d in DOMAINS]))
        row={'method':name,'epoch':epoch,'mean_source_macro_f1':mean_f1,**totals,
             **{f'{d}_{k}':v for d in DOMAINS for k,v in scores[d].items()}}
        rows.append(row);pd.DataFrame(rows).to_csv(history_file,index=False)
        print(f'{name}: epoch {epoch} | mean source F1 {mean_f1:.4f} | '
              f'CE {totals["classification"]:.4f} | MMD {totals["mmd"]:.4f} | '
              f'feature norm {totals["median_feature_norm"]:.1f} | '
              f'max unclipped grad {totals["max_gradient_before_clip"]:.1f}')
        if mean_f1>best+1e-12:
            best=mean_f1;stale=0
            torch.save({'model':net.state_dict(),'method':name,'strength':strength,
                        'epoch':epoch,'source_validation':scores,'seed':SEED},checkpoint)
        else:stale+=1
        if stale>=5:break
    done.write_text(json.dumps({'epochs_ran':len(rows),'best_source_macro_f1':best}))
    print(name,'selected at source epoch',torch.load(checkpoint,map_location='cpu',weights_only=True)['epoch'])

for lam in STRENGTHS:train(f'DAN-DG {lam:g}',lam)
train('SAM')
print('Source-only ERM was reused; four new runs completed. No Sketch image opened.')

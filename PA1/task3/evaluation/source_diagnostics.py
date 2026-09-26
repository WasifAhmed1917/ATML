# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 9
rng=np.random.default_rng(SEED)
probe_paths={d:source_val[d].paths for d in DOMAINS}
probe_n=min(map(len,probe_paths.values()))
probe_idx={d:rng.choice(len(probe_paths[d]),size=probe_n,replace=False) for d in DOMAINS}
fixed_rng=np.random.default_rng(SEED)
fixed_idx={d:fixed_rng.choice(len(source_val[d]),32,replace=False) for d in DOMAINS}
fixed_images=[];fixed_labels=[]
for d in DOMAINS:
    for i in fixed_idx[d]:
        x,y=source_val[d][int(i)];fixed_images.append(x);fixed_labels.append(y)
fixed_x=torch.stack(fixed_images).to(DEVICE)
fixed_y=torch.tensor(fixed_labels,dtype=torch.long,device=DEVICE)
assert len(fixed_y)==96

@torch.inference_mode()
def source_features(net,domain,indices):
    net.eval();ds=Images([probe_paths[domain][int(i)] for i in indices],eval_tf)
    chunks=[]
    for x,_ in loader(ds,64):
        f,_=net(x.to(DEVICE,non_blocking=True));chunks.append(f.cpu().numpy())
    return np.concatenate(chunks)

def sharpness(net):
    net.eval();params=[p for p in net.parameters() if p.requires_grad]
    net.zero_grad(set_to_none=True)
    _,logits=net(fixed_x);base=F.cross_entropy(logits,fixed_y)
    base.backward()
    with torch.no_grad():
        gnorm=torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]))
        assert bool(torch.isfinite(gnorm))
        eps={p:0.05*p.grad.detach().clone()/gnorm.clamp_min(1e-12)
             for p in params if p.grad is not None}
        for p,e in eps.items():p.add_(e)
    try:
        with torch.no_grad():
            _,perturbed_logits=net(fixed_x)
            perturbed=F.cross_entropy(perturbed_logits,fixed_y)
            delta=float((perturbed-base.detach()).cpu())
    finally:
        with torch.no_grad():
            for p,e in eps.items():p.sub_(e)
        net.zero_grad(set_to_none=True)
    return delta

diagnostic_rows=[]
for name in NAMES:
    net,_=selected(name)
    X=np.concatenate([source_features(net,d,probe_idx[d]) for d in DOMAINS])
    domains=np.concatenate([np.full(probe_n,i,dtype=int) for i,_ in enumerate(DOMAINS)])
    tr,te=train_test_split(np.arange(len(domains)),test_size=.3,random_state=SEED,stratify=domains)
    probe=LogisticRegression(C=1.,max_iter=2000,random_state=SEED)
    probe.fit(X[tr],domains[tr])
    diagnostic_rows.append({'method':name,'validation_examples_per_domain':probe_n,
                            'source_domain_separability':accuracy_score(domains[te],probe.predict(X[te])),
                            'sharpness_increase':sharpness(net)})
diagnostics=pd.DataFrame(diagnostic_rows)
diagnostics.to_csv(OUT/'source_diagnostics.csv',index=False)
display(diagnostics.round(5))
print('Source models and diagnostics fixed. Sketch has not been opened by this notebook.')

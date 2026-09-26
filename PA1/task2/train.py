# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 10
@torch.inference_mode()
def predict_source(net,dataset):
    net.eval(); ys=[]; ps=[]
    for x,y in loader(dataset,64):
        _,logits=net(x.to(DEVICE,non_blocking=True));ys.extend(y.numpy());ps.extend(logits.argmax(1).cpu().numpy())
    return np.asarray(ys),np.asarray(ps)
def source_metrics(net):
    result={}
    for d in DOMAINS:
        y,p=predict_source(net,source_val[d])
        result[d]={'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=range(7),average='macro',zero_division=0)}
    return result

def run(name,strength):
    safe=name.lower().replace(' ','_').replace('.','p').replace('-','_')
    best_file=OUT/f'{safe}.pt'; hist_file=OUT/f'{safe}_history.csv'
    if (OUT/f'{safe}_done.json').exists() and best_file.exists() and hist_file.exists():
        print(name,'already complete; loading checkpoint')
        return
    seed_all(SEED);net=make_model();net.load_state_dict(torch.load(initial_file,map_location=DEVICE,weights_only=True))
    discriminator=None
    if name in ('DANN','CDAN'):
        discriminator=Discriminator(512*(7 if name=='CDAN' else 1)).to(DEVICE)
        nn.init.zeros_(discriminator[-1].weight)
        nn.init.zeros_(discriminator[-1].bias)
    params=list(net.parameters())+(list(discriminator.parameters()) if discriminator is not None else [])
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    steps_per_epoch=max(math.ceil(len(source_train[d])/8) for d in DOMAINS)
    full_steps=30*steps_per_epoch
    best=-float('inf');stale=0;rows=[]
    for epoch in range(1,31):
        net.train();freeze_bn_stats(net)
        if discriminator is not None:discriminator.train()
        src,tgt=epoch_loaders(epoch)
        src_it={d:batches_forever(src[d]) for d in DOMAINS}
        tgt_it=batches_forever(tgt) if name!='Source-only' else None
        totals={'classification':0.,'alignment_or_domain':0.,'domain_accuracy':0.,
                'median_feature_norm':0.,'max_gradient_before_clip':0.}
        for step in tqdm(range(steps_per_epoch),desc=f'{name} epoch {epoch}',leave=False):
            xs=[];ys=[]
            for d in DOMAINS:
                x,y=next(src_it[d]);xs.append(x);ys.append(y)
            x=torch.cat(xs).to(DEVICE,non_blocking=True)
            y=torch.cat(ys).to(DEVICE,non_blocking=True)
            if tgt_it is not None:
                t=next(tgt_it).to(DEVICE,non_blocking=True)
                features,logits=net(torch.cat([x,t]))
                sf,tf=features[:24],features[24:]
                source_logits=logits[:24]
            else:
                sf,source_logits=net(x)
            cls=F.cross_entropy(source_logits,y)
            alignment=cls.new_zeros(());dom_acc=0.
            if name.startswith('DAN '):alignment=mmd_rbf(sf,tf)
            elif name in ('DANN','CDAN'):
                p=( (epoch-1)*steps_per_epoch+step )/max(1,full_steps-1)
                alpha=2/(1+math.exp(-10*p))-1
                joint=domain_input(name,features,logits)
                domain_y=torch.cat([torch.zeros(24,dtype=torch.long,device=DEVICE),
                                    torch.ones(24,dtype=torch.long,device=DEVICE)])
                domain_logits=discriminator(Reverse.apply(joint,alpha))
                alignment=F.cross_entropy(domain_logits,domain_y)
                dom_acc=(domain_logits.argmax(1)==domain_y).float().mean().item()
            loss=cls+(strength*alignment if name.startswith('DAN ') else alignment)
            assert bool(torch.isfinite(loss)),f'Nonfinite loss in {name} epoch {epoch}'
            opt.zero_grad(set_to_none=True);loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            opt.step()
            norm=float(sf.detach().norm(dim=1).median())
            if not math.isfinite(norm) or norm>1e4:
                raise RuntimeError(f'{name} epoch {epoch}: unstable source feature norm {norm:.3g}')
            totals['median_feature_norm']+=norm/steps_per_epoch
            totals['max_gradient_before_clip']=max(totals['max_gradient_before_clip'],float(grad_norm))
            totals['classification']+=cls.item()/steps_per_epoch
            totals['alignment_or_domain']+=alignment.item()/steps_per_epoch
            totals['domain_accuracy']+=dom_acc/steps_per_epoch
        scores=source_metrics(net)
        mean_f1=float(np.mean([scores[d]['macro_f1'] for d in DOMAINS]))
        row={'method':name,'epoch':epoch,'mean_source_macro_f1':mean_f1,
             **totals,**{f'{d}_{k}':v for d in DOMAINS for k,v in scores[d].items()}}
        rows.append(row);pd.DataFrame(rows).to_csv(hist_file,index=False)
        print(f'{name}: epoch {epoch} | mean source macro-F1 {mean_f1:.4f} | class {totals["classification"]:.4f} | align/domain {totals["alignment_or_domain"]:.4f} | feature norm {totals["median_feature_norm"]:.1f} | max grad {totals["max_gradient_before_clip"]:.1f}')
        if mean_f1>best+1e-12:
            best=mean_f1;stale=0
            torch.save({'model':net.state_dict(),'method':name,'strength':strength,'epoch':epoch,
                        'source_validation':scores,'seed':SEED,
                        'discriminator':discriminator.state_dict() if discriminator is not None else None},best_file)
        else:stale+=1
        if stale>=5:break
    (OUT/f'{safe}_done.json').write_text(json.dumps({'epochs_ran':len(rows),'best_source_macro_f1':best}))
    print(name,'best source epoch',torch.load(best_file,map_location='cpu',weights_only=True)['epoch'])

for name,strength in RUNS:run(name,strength)
shutil.copy2(OUT/'source_only.pt',OUT/'source_only_erm.pt')
assert all((OUT/(name.lower().replace(' ','_').replace('.','p').replace('-','_')+'.pt')).exists() for name,_ in RUNS)
print('All six runs complete; target labels have not been read.')

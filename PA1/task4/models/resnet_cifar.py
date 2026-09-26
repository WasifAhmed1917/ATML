# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 5
class CifarResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.net=torchvision.models.resnet18(weights=None,num_classes=num_classes)
        self.net.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.net.maxpool=nn.Identity()
    def pre(self,x):
        n=self.net
        x=n.relu(n.bn1(n.conv1(x)))
        return n.layer2(n.layer1(x))
    def post_features(self,h):
        n=self.net
        return torch.flatten(n.avgpool(n.layer4(n.layer3(h))),1)
    def features(self,x): return self.post_features(self.pre(x))
    def forward(self,x): return self.net.fc(self.features(x))

probe=CifarResNet18().to(DEVICE).eval()
with torch.no_grad():
    assert probe.features(torch.zeros(2,3,32,32,device=DEVICE)).shape==(2,512)
    assert probe(torch.zeros(2,3,32,32,device=DEVICE)).shape==(2,10)
del probe

def evaluate_accuracy(net,ds):
    net.eval(); correct=total=0
    with torch.inference_mode():
        for x,y,_ in loader(ds):
            pred=net(x.to(DEVICE,non_blocking=True))[:,:10].argmax(1)
            correct+=(pred==y.to(DEVICE,non_blocking=True)).sum().item(); total+=len(y)
    return correct/total

def fit(name,epochs,lr,aug,init_from=None):
    path=MODELS/f'{name}_best.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['epochs']==epochs and ck['seed']==SEED and ck['lr']==lr
        net=CifarResNet18(15 if name=='PROSER' else 10).to(DEVICE)
        net.load_state_dict(ck['state']); print(name,'reused best checkpoint (epoch',ck['epoch'],')')
        return net,ck
    set_seed()
    net=CifarResNet18().to(DEVICE)
    if name=='PROSER':
        net.load_state_dict(torch.load(init_from,map_location='cpu',weights_only=False)['state'])
        old=net.net.fc
        new=nn.Linear(old.in_features,15).to(DEVICE)
        with torch.no_grad():
            new.weight[:10].copy_(old.weight); new.bias[:10].copy_(old.bias)
        net.net.fc=new
    optim=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs)
    train_ds=Indexed(raw,train_ids,aug)
    best=-1.0; ck=None
    for epoch in range(1,epochs+1):
        net.train(); running=np.zeros(3,dtype=float); batches=0
        for x,y,_ in loader(train_ds,shuffle=True,epoch=epoch):
            x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if name=='PROSER':
                assert len(x)>=2
                n=len(x)//2

                z=net(x[:n]); z_known=z[:,:10]; z_dummy=z[:,10:].max(1,keepdim=True).values
                full=torch.cat([z_known,z_dummy],dim=1)
                masked=full.clone(); masked[torch.arange(n,device=DEVICE),y[:n]]=-1e9
                loss_known=F.cross_entropy(full,y[:n])
                loss_masked=F.cross_entropy(masked,torch.full_like(y[:n],10))

                mix_x=x[n:]; mix_y=y[n:]
                choices=(mix_y[:,None]!=mix_y[None,:]).float()
                assert (choices.sum(1)>0).all(), 'Second batch half needs at least two classes.'
                partners=torch.multinomial(choices,1).flatten()
                assert torch.all(mix_y!=mix_y[partners])
                h=net.pre(mix_x)
                lam=np.random.beta(2,2)
                f=net.post_features(lam*h+(1-lam)*h[partners])
                zm=net.net.fc(f)
                mixed=torch.cat([zm[:,:10],zm[:,10:].max(1,keepdim=True).values],dim=1)
                loss_mix=F.cross_entropy(mixed,torch.full((len(mix_x),),10,device=DEVICE,dtype=torch.long))
                loss=loss_known+loss_masked+0.1*loss_mix
                running+=np.array([loss_known.item(),loss_masked.item(),loss_mix.item()])
            else:
                loss=F.cross_entropy(net(x),y)
                running[0]+=loss.item()
            assert torch.isfinite(loss).item(),f'{name}: nonfinite loss in epoch {epoch}'
            loss.backward(); optim.step(); batches+=1
        sched.step()
        acc=evaluate_accuracy(net,known_val)
        if acc>best:
            best=acc
            ck={'state':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                'epoch':epoch,'val_accuracy':acc,'seed':SEED,'lr':lr,'epochs':epochs,
                'optimizer':'SGD(momentum=0.9,weight_decay=5e-4)', 'schedule':'cosine',
                'batch_size':128,'augmentation':name,'loss':(running/batches).tolist()}
            torch.save(ck,path)
        if epoch==1 or epoch%10==0 or epoch==epochs:
            print(f'{name:7s} epoch {epoch:3d}/{epochs} | val acc {acc:.4f} | best {best:.4f} | losses {np.round(running/batches,4)}')
    net.load_state_dict(ck['state'])
    print(f'{name}: selected epoch {ck["epoch"]}, validation accuracy {best:.4f}')
    return net,ck

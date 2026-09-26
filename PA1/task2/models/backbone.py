# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 6
weights=ResNet18_Weights.IMAGENET1K_V1
mean,std=weights.transforms().mean,weights.transforms().std
train_tf=transforms.Compose([transforms.Resize((256,256)),transforms.RandomCrop(224),
                              transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),
                             transforms.ToTensor(),transforms.Normalize(mean,std)])
class Images(Dataset):
    def __init__(self,relpaths,transform,labeled=True):
        self.paths=list(relpaths); self.transform=transform; self.labeled=labeled
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(ROOT/self.paths[i]) as im: x=self.transform(im.convert('RGB'))
        if self.labeled:
            y=CLASSES.index(Path(self.paths[i]).parent.name.lower())
            return x,y
        return x
source_train={d:Images(splits[d]['train'],train_tf) for d in DOMAINS}
source_val={d:Images(splits[d]['val'],eval_tf) for d in DOMAINS}
target_train=Images(target_rel,train_tf,labeled=False)
target_diagnostic=Images(target_rel,eval_tf,labeled=False)
def loader(ds,batch,shuffle=False,seed=SEED,drop_last=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,drop_last=drop_last,
                      num_workers=2,pin_memory=True,persistent_workers=False,
                      generator=torch.Generator().manual_seed(seed))
def epoch_loaders(epoch):
    src={d:loader(source_train[d],8,True,SEED+1000*epoch+i,True)
         for i,d in enumerate(DOMAINS)}
    tgt=loader(target_train,24,True,SEED+1000*epoch+100,True)
    return src,tgt
def batches_forever(dl):
    while True:yield from dl
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet18(weights=weights)
        self.backbone.fc=nn.Identity()
        self.classifier=nn.Linear(512,7)
    def forward(self,x):
        f=self.backbone(x)
        return f,self.classifier(f)
def freeze_bn_stats(net):
    for module in net.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()

def make_model():
    seed_all(SEED)
    model=Net().to(DEVICE)
    model.train(); freeze_bn_stats(model)
    assert all(not m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
    assert all(m.weight.requires_grad for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
    return model
# Cache the one common initialization; this checkpoint never uses PACS images.
initial_file=OUT/'imagenet_initial.pt'
if not initial_file.exists():torch.save(make_model().state_dict(),initial_file)
print('Transform:',train_tf,'\nValidation:',eval_tf)

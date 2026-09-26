# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 3
HF_REPO='Azeez577/PACS'
HF_REVISION='46a0a83'
HF_SHA256='42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147'
assert t2config['dataset_sha256']==HF_SHA256
extracted=Path('/content/PACS_extracted')
if not (extracted/'PACS').is_dir():
    archive=Path('/content/PACS.zip')
    if not archive.is_file():
        subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub'],check=True)
        from huggingface_hub import hf_hub_download
        archive=Path(hf_hub_download(repo_id=HF_REPO,repo_type='dataset',
                                     filename='PACS.zip',revision=HF_REVISION,local_dir='/content'))
    assert zipfile.is_zipfile(archive),'PACS archive missing or damaged.'
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==HF_SHA256,'PACS archive differs from Task 2.'
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            assert (extracted/info.filename).resolve().is_relative_to(extracted.resolve()),'Unsafe archive member.'
        z.extractall(extracted)
ROOT=extracted/'PACS'
assert all((ROOT/d).is_dir() for d in DOMAINS),'Expected Task 2 PACS source directories.'
split_file=TASK2/'splits/pacs_sketch_seed6304.json'
splits=json.loads(split_file.read_text())
assert set(splits)==set(DOMAINS)
extensions={'.jpg','.jpeg','.png','.bmp','.webp'}
for d in DOMAINS:
    found={str(p.relative_to(ROOT)) for p in (ROOT/d).rglob('*')
           if p.is_file() and p.suffix.lower() in extensions}
    train=set(splits[d]['train']); val=set(splits[d]['val'])
    assert not train&val and train|val==found,(d,'Task 2 split/data mismatch')
    assert all(Path(p).parts[0]==d and Path(p).parent.name.lower() in CLASSES for p in found)
    print(d,'train',len(train),'validation',len(val))

weights=ResNet18_Weights.IMAGENET1K_V1
mean,std=weights.transforms().mean,weights.transforms().std
train_tf=transforms.Compose([transforms.Resize((256,256)),transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                             transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),
                            transforms.ToTensor(),transforms.Normalize(mean,std)])
class Images(Dataset):
    def __init__(self,paths,transform):self.paths=list(paths);self.transform=transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        path=self.paths[i]
        with Image.open(ROOT/path) as im:x=self.transform(im.convert('RGB'))
        return x,CLASSES.index(Path(path).parent.name.lower())
source_train={d:Images(splits[d]['train'],train_tf) for d in DOMAINS}
source_val={d:Images(splits[d]['val'],eval_tf) for d in DOMAINS}
def loader(ds,batch,shuffle=False,seed=SEED,drop_last=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,drop_last=drop_last,
                      num_workers=2,pin_memory=True,persistent_workers=False,
                      generator=torch.Generator().manual_seed(seed))
def epoch_loaders(epoch):
    return {d:loader(source_train[d],8,True,SEED+1000*epoch+i,True)
            for i,d in enumerate(DOMAINS)}
def forever(dl):
    while True:yield from dl
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=resnet18(weights=weights)
        self.backbone.fc=nn.Identity();self.classifier=nn.Linear(512,7)
    def forward(self,x):
        f=self.backbone(x)
        return f,self.classifier(f)
def freeze_bn_stats(net):
    for m in net.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
def initial_net():
    seed_all(SEED)
    net=Net().to(DEVICE)
    net.load_state_dict(torch.load(TASK2/'imagenet_initial.pt',map_location=DEVICE,weights_only=True))
    net.train();freeze_bn_stats(net)
    return net
erm_checkpoint=torch.load(TASK2/'source_only_erm.pt',map_location=DEVICE,weights_only=True)
assert erm_checkpoint['seed']==SEED and erm_checkpoint['method']=='Source-only'
print('Reused ERM selected at Task 2 epoch',erm_checkpoint['epoch'])

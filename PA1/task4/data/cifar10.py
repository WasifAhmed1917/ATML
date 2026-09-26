# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 3
raw=torchvision.datasets.CIFAR10('/content/data',train=True,download=True)
known_test=torchvision.datasets.CIFAR10('/content/data',train=False,download=True)
assert len(raw)==50000 and len(known_test)==10000
train_ids,val_ids=train_test_split(np.arange(50000),test_size=0.1,stratify=raw.targets,random_state=SEED)
train_ids=np.sort(train_ids); val_ids=np.sort(val_ids)
assert len(train_ids)==45000 and len(val_ids)==5000 and not set(train_ids)&set(val_ids)
assert np.bincount(np.asarray(raw.targets)[train_ids],minlength=10).tolist()==[4500]*10
assert np.bincount(np.asarray(raw.targets)[val_ids],minlength=10).tolist()==[500]*10
np.savez(RESULTS/'cifar10_split_seed6304.npz',train_ids=train_ids,val_ids=val_ids)

sums=np.zeros(3,np.float64); sq=np.zeros(3,np.float64)
for block in np.array_split(train_ids,90):
    v=raw.data[block].astype(np.float64)/255.0
    sums+=v.sum((0,1,2)); sq+=(v*v).sum((0,1,2))
pixels=len(train_ids)*32*32
mean=sums/pixels; std=np.sqrt(sq/pixels-mean**2)
assert np.all(std>0)
normalize=T.Normalize(tuple(mean.tolist()),tuple(std.tolist()))
base_aug=[T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()]
vanilla_aug=T.Compose(base_aug+[T.ToTensor(),normalize])
gcsc_aug=T.Compose(base_aug+[T.RandAugment(num_ops=2,magnitude=9),T.ToTensor(),normalize])
clean_transform=T.Compose([T.ToTensor(),normalize])

class Indexed(Dataset):
    def __init__(self,base,ids,transform):
        self.base=base; self.ids=np.asarray(ids,dtype=np.int64); self.transform=transform
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        idx=int(self.ids[i]); im,label=self.base[idx]
        return self.transform(im),int(label),idx

def loader(ds,shuffle=False,epoch=0):
    g=torch.Generator().manual_seed(SEED+epoch)
    return DataLoader(ds,batch_size=128,shuffle=shuffle,num_workers=2,pin_memory=True,
                      persistent_workers=False,generator=g)

known_train_clean=Indexed(raw,train_ids,clean_transform)
known_val=Indexed(raw,val_ids,clean_transform)
known_test_clean=Indexed(known_test,np.arange(len(known_test)),clean_transform)
classes=raw.classes
print('CIFAR-10 classes:',classes)
print('Train / validation / test:',len(train_ids),len(val_ids),len(known_test),'| train mean:',np.round(mean,5),'| std:',np.round(std,5))

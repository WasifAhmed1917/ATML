# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 6
train_ds=STL10(root=str(DATA),split='train',download=True)
test_ds=STL10(root=str(DATA),split='test',download=True)
assert list(train_ds.classes)==CLASSES, (train_ds.classes,CLASSES)
y_train=np.asarray(train_ds.labels,dtype=np.int64); y_test=np.asarray(test_ds.labels,dtype=np.int64)
tr_idx,va_idx=train_test_split(np.arange(len(train_ds)),test_size=0.2,stratify=y_train,random_state=SEED)
rng=np.random.default_rng(SEED)
selected=[]; shortage={}
for c in range(10):
    pool=np.flatnonzero(y_test==c)
    k=min(50,len(pool)); shortage[CLASSES[c]]=50-k
    selected.extend(rng.choice(pool,k,replace=False).tolist())
selected=np.asarray(selected,dtype=np.int64)
assert len(selected)==len(set(selected))
assert all(y_test[selected].tolist().count(c)==min(50,int(sum(y_test==c))) for c in range(10))
pd.DataFrame({'selected_order':np.arange(len(selected)), 'official_test_index':selected,
              'image_id':['STL10-test-%05d'%i for i in selected], 'class_id':y_test[selected],
              'class_name':[CLASSES[c] for c in y_test[selected]]}).to_csv(OUT/'selected_test_ids.csv',index=False)
(OUT/'train_val_ids.json').write_text(json.dumps({'train_official_indices':tr_idx.tolist(),'val_official_indices':va_idx.tolist()}))
print('Train / validation / test subset:',len(tr_idx),len(va_idx),len(selected))
print('Test counts:',pd.Series(y_test[selected]).value_counts().sort_index().rename(index=dict(enumerate(CLASSES))).to_dict())
print('Shortfall from 50 per class:',shortage)

def common(image):
    return np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),dtype=np.uint8)
def subset_images(ds, ids):
    return np.stack([common(ds[int(i)][0]) for i in tqdm(ids,desc='Resize')])
Xtr=subset_images(train_ds,tr_idx); Xva=subset_images(train_ds,va_idx)
X=subset_images(test_ds,selected); y=y_test[selected]
assert X.shape[1:]==(224,224,3) and X.dtype==np.uint8
np.savez_compressed(OUT/'clean_subset.npz',images=X,official_indices=selected,labels=y)
print('Common pixels:',X.shape)

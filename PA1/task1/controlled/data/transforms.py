# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 23
def translate(im,delta,direction):

    if delta==0: return im.copy()
    p=np.pad(im,((delta,delta),(delta,delta),(0,0)),mode='reflect')
    sy,sx={'right':(delta,0),'left':(delta,2*delta),
           'down':(0,delta),'up':(2*delta,delta)}[direction]
    return np.ascontiguousarray(p[sy:sy+224,sx:sx+224])
DIRECTIONS=['right','left','down','up']
assert np.array_equal(translate(X[0],0,'right'),X[0])
def shuffle_patches(im,permutation):
    patches=im.reshape(4,56,4,56,3).transpose(0,2,1,3,4).reshape(16,56,56,3)
    return np.ascontiguousarray(patches[permutation].reshape(4,4,56,56,3).transpose(0,2,1,3,4).reshape(224,224,3))
rng=np.random.default_rng(SEED); perms=[]
for _ in range(len(X)):
    p=rng.permutation(16)
    while np.array_equal(p,np.arange(16)): p=rng.permutation(16)
    perms.append(p)
perms=np.array(perms)
PATCH=np.stack([shuffle_patches(im,p) for im,p in zip(X,perms)])
assert all(np.array_equal(np.sort(PATCH[i].reshape(-1,3),axis=0),np.sort(X[i].reshape(-1,3),axis=0)) for i in range(2))
np.save(OUT/'patch_permutations.npy',perms)
fig,ax=plt.subplots(1,3,figsize=(8,3))
for a,z,title in zip(ax,[X[0],translate(X[0],32,'right'),PATCH[0]],['Clean','Right 32 px','4×4 shuffled']):
    a.imshow(z); a.set_title(title); a.axis('off')
plt.show()

# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 13
def grayscale(im):
    return np.asarray(ImageOps.grayscale(Image.fromarray(im)).convert('RGB'),dtype=np.uint8)
def hue_rotate(im):
    hsv=np.asarray(Image.fromarray(im).convert('HSV')).copy()
    hsv[:,:,0]=(hsv[:,:,0].astype(np.uint16)+round(HUE_DEGREES*256/360))%256
    return np.asarray(Image.fromarray(hsv,'HSV').convert('RGB'),dtype=np.uint8)
GRAY=np.stack([grayscale(im) for im in tqdm(X,desc='Grayscale')])
HUE=np.stack([hue_rotate(im) for im in tqdm(X,desc='Hue')])
assert GRAY.shape==HUE.shape==X.shape
fig,ax=plt.subplots(3,3,figsize=(7,7))
for j,img in enumerate([X,GRAY,HUE]):
    for k in range(3): ax[j,k].imshow(img[k]); ax[j,k].axis('off')
for j,label in enumerate(['Clean','Grayscale','Hue +90°']): ax[j,0].set_ylabel(label)
plt.tight_layout(); plt.show()

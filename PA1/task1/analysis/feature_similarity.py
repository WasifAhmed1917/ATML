# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 31
def cosine_mean(a,b):
    a=F.normalize(torch.from_numpy(a).float(),dim=1)
    b=F.normalize(torch.from_numpy(b).float(),dim=1)
    return (a*b).sum(1).mean().item()
STABILITY=[]
for m in ['ResNet-50','ViT-B/16','CLIP head']:
    fc=CLEAN_FEATURES[m]
    for name,z in [('grayscale',COLOR_FEATURES[m]['grayscale']),('cue conflict',CUE_FEATURES[m]),
                   ('patch shuffle',PATCH_FEATURES[m])]:
        base=fc[CUE_CONTENT] if name=='cue conflict' else fc
        STABILITY.append(dict(backbone=m,intervention=name,n=len(z),mean_cosine=cosine_mean(base,z)))
    for d,z in TRANS16[m].items():
        STABILITY.append(dict(backbone=m,intervention=f'translation 16 px {d}',n=len(z),mean_cosine=cosine_mean(fc,z)))
    STABILITY.append(dict(backbone=m,intervention='translation 16 px, four-direction mean',n=4*len(fc),
                          mean_cosine=np.mean([cosine_mean(fc,z) for z in TRANS16[m].values()])))
stability=pd.DataFrame(STABILITY);stability.to_csv(OUT/'representation_stability.csv',index=False)
show_table(stability.round(4))

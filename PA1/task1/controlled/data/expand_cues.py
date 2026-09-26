# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 20
review=pd.read_csv(review_file,dtype={'accept':'string'},keep_default_na=False)
assert len(review)==len(CAND)
STRATA=[pair for a,b in PAIRS for pair in [(a,b),(b,a)]]
exhausted=[(a,b) for a,b in STRATA if
    (review.loc[(review.content_class==a)&(review.style_class==b),'accept']=='1').sum()<20 and
    not (review.loc[(review.content_class==a)&(review.style_class==b),'accept']=='').any()]
assert exhausted, 'No exhausted direction: continue visual review.'
rng=np.random.default_rng(SEED+len(CAND))
more=[];more_masks=[];new_rows=[]
for ca,cb in exhausted:
    sources=np.flatnonzero(y==CLASSES.index(ca)); donors=np.flatnonzero(y==CLASSES.index(cb))
    group=review[(review.content_class==ca)&(review.style_class==cb)]
    used=set(zip(group.content_id.astype(int),group.style_id.astype(int)))
    for _ in tqdm(range(40),desc=f'Expand {ca} → {cb}'):
        while True:
            ci=int(rng.choice(sources));si=int(rng.choice(donors))
            if (ci,si) not in used: break
        used.add((ci,si)); image,mask,_=stylize(X[ci],X[si])
        more.append(image);more_masks.append(mask)
        new_rows.append(dict(candidate_id=len(CAND)+len(more)-1,content_id=ci,style_id=si,
                             content_class=ca,style_class=cb,accept=''))
CAND=np.concatenate((CAND,np.stack(more)))
MASKS=np.concatenate((MASKS,np.stack(more_masks)))
review=pd.concat([review,pd.DataFrame(new_rows)],ignore_index=True)

np.savez_compressed(parts/f'expanded_{len(CAND)}.npz',images=np.stack(more),masks=np.stack(more_masks))
review.to_csv(review_file,index=False)
log=OUT/'cue_expansion_log.csv'
pd.concat([pd.read_csv(log),pd.DataFrame(new_rows)] if log.exists() else [pd.DataFrame(new_rows)],ignore_index=True).to_csv(log,index=False)
print('Added',len(more),'candidates. Rerun the reviewer cell.')

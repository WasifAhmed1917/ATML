# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 21

review=pd.read_csv(review_file,dtype={'accept':'string'},keep_default_na=False)
assert review['candidate_id'].tolist()==list(range(len(CAND)))
assert set(review['accept'])<= {'','0','1'}, 'Only blank, 0, or 1 are valid review decisions.'
counts=review.assign(accept=review['accept'].replace({'':'unreviewed','0':'rejected','1':'accepted'})).groupby(['content_class','style_class','accept']).size().rename('count').reset_index()
display(counts)
accepted_all=review[review['accept']=='1'].copy()
stratum_counts=accepted_all.groupby(['content_class','style_class']).size()
assert len(stratum_counts)==10 and stratum_counts.min()>=20, (
    f'Need >=20 visually valid conflicts in every pair/direction; got {stratum_counts.to_dict()}. '
    'Revise generation and repeat blinded review before evaluating.')

accepted=accepted_all.groupby(['content_class','style_class'],sort=False).head(20).reset_index(drop=True)
assert len(accepted)==200 and accepted.groupby(['content_class','style_class']).size().eq(20).all()
ACCEPTED_CANDIDATE=accepted.candidate_id.to_numpy(dtype=int)
CUE=CAND[ACCEPTED_CANDIDATE]
CUE_CONTENT=accepted.content_id.to_numpy(dtype=int)
CUE_SHAPE=np.array([CLASSES.index(s) for s in accepted.content_class]); CUE_TEXTURE=np.array([CLASSES.index(s) for s in accepted.style_class])
assert np.all(y[CUE_CONTENT]==CUE_SHAPE) and np.all(CUE_SHAPE!=CUE_TEXTURE)
accepted.to_csv(OUT/'cue_accepted_ids.csv',index=False)
counts.to_csv(OUT/'cue_review_counts.csv',index=False)
print(f'Accepted: {len(accepted_all)}; rejected: {(review.accept=='0').sum()}; unreviewed: {(review.accept=='').sum()}; balanced evaluation: {len(accepted)}.')

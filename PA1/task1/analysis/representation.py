# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 32
rng=np.random.default_rng(SEED)
vis_rows=rng.choice(np.arange(len(accepted)),VIS_N,replace=False)
vis_ids=CUE_CONTENT[vis_rows]; vis_labels=y[vis_ids]
assert len(np.unique(vis_ids))==VIS_N, 'Expected unique content images (disjoint class pairs).'
np.savez(OUT/'visualization_ids.npz',content_id=vis_ids,candidate_id=ACCEPTED_CANDIDATE[vis_rows],labels=vis_labels)
for m in ['ResNet-50','ViT-B/16','CLIP head']:
    conditions={'clean':CLEAN_FEATURES[m][vis_ids],
                'grayscale':COLOR_FEATURES[m]['grayscale'][vis_ids],
                'cue conflict':CUE_FEATURES[m][vis_rows],
                'translation right 16 px':TRANS16[m]['right'][vis_ids],
                'patch shuffle':PATCH_FEATURES[m][vis_ids]}
    Z=np.concatenate(list(conditions.values())).astype(np.float32)
    Z=F.normalize(torch.from_numpy(Z),dim=1).numpy()
    emb=TSNE(n_components=2,perplexity=30,metric='cosine',init='pca',learning_rate='auto',
             random_state=SEED).fit_transform(Z)
    fig,ax=plt.subplots(figsize=(10,8)); markers=['o','s','^','D','X'];
    cmap=plt.get_cmap('tab10')
    for ci,(condition,marker) in enumerate(zip(conditions,markers)):
        points=emb[ci*VIS_N:(ci+1)*VIS_N]
        ax.scatter(points[:,0],points[:,1],c=[cmap(int(v)) for v in vis_labels],
                   s=28,marker=marker,alpha=.78,label=condition,edgecolors='none')
    handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=cmap(i),markersize=8,label=label) for i,label in enumerate(CLASSES)]
    leg=ax.legend(handles=handles,title='Ground-truth class',bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)
    ax.add_artist(leg);ax.legend(title='Condition / marker',bbox_to_anchor=(1.01,.48),loc='upper left',fontsize=8)
    ax.set(title=f'{m} · joint t-SNE',xlabel='t-SNE 1',ylabel='t-SNE 2')
    plt.tight_layout();plt.savefig(OUT/(m.replace('/','_').replace(' ','_')+'_tsne.png'),dpi=170,bbox_inches='tight');plt.show()
    pd.DataFrame({'x':emb[:,0],'y':emb[:,1],'condition':np.repeat(list(conditions),VIS_N),
                  'content_id':np.tile(vis_ids,len(conditions)),'class_id':np.tile(vis_labels,len(conditions))}).to_csv(
                      OUT/(m.replace('/','_').replace(' ','_')+'_tsne.csv'),index=False)
print('Settings: t-SNE, cosine metric, normalized input vectors, perplexity 30, PCA init, learning rate auto, seed 6304; 100 exact matched images per condition. Projection coordinates across backbones are not comparable.')

# %% Original notebook cell 33
per_class = pd.DataFrame({
    'class': CLASSES,
    'n': [int((y == c).sum()) for c in range(10)],
    **{
        model: [
            float((PRED_CLEAN[model][y == c] == c).mean())
            for c in range(10)
        ]
        for model in NAMES
    },
})
display(per_class.round(3))
per_class.to_csv(OUT / 'per_class_clean_accuracy.csv', index=False)

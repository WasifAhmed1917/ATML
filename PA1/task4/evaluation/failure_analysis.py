# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 13
fig,axes=plt.subplots(1,3,figsize=(14,3.5),layout='constrained')
for ax,s in zip(axes,['MSP','MLS','Mahalanobis']):
    for label,values,color in [('Known test',vscore_test[s],'#2980b9'),('Near',vscore_near[s],'#e67e22'),('Far',vscore_far[s],'#8e44ad')]:
        ax.hist(values,bins=45,density=True,histtype='step',linewidth=1.5,color=color,label=label)
    ax.axvline(thresholds['Vanilla '+s],color='black',linestyle='--',linewidth=1,label='Known-val cutoff')
    ax.set_title(s); ax.set_xlabel('Unknownness (higher = more novel)');ax.set_ylabel('Density')
axes[0].legend(fontsize=8)
fig.savefig(RESULTS/'vanilla_score_distributions.png',dpi=180,bbox_inches='tight')
plt.show()

failure_rows=[]; examples={}
for group,arr,vals in [('Near',vnear,vscore_near['MLS']),('Far',vfar,vscore_far['MLS'])]:
    accepted=np.flatnonzero(vals<=thresholds['Vanilla MLS'])
    assert len(accepted)>=3,f'Fewer than three accepted {group} unknowns; report actual count rather than replacing threshold.'
    classes100=np.asarray([c100.classes[int(c)] for c in arr['labels']])
    preds=arr['logits'][:,:10].argmax(1)
    for klass in sorted(set(classes100)):
        positions=np.flatnonzero(classes100==klass)
        taken=positions[vals[positions]<=thresholds['Vanilla MLS']]
        if len(taken):
            counts=pd.Series([classes[int(p)] for p in preds[taken]]).value_counts()
            failure_rows.append({'Group':group,'Unknown class':klass,'Accepted / total':f'{len(taken)} / {len(positions)}',
                                 'Acceptance rate':len(taken)/len(positions),'Most frequent known prediction':counts.index[0]})

    selected=[]
    for klass in sorted(set(classes100[accepted])):
        candidates=accepted[classes100[accepted]==klass]
        selected.append(int(candidates[np.argmin(vals[candidates])]))
    selected=sorted(selected,key=lambda i:vals[i])[:3]
    if len(selected)<3:
        for idx in accepted[np.argsort(vals[accepted])]:
            if int(idx) not in selected: selected.append(int(idx))
            if len(selected)==3: break
    assert len(selected)==3
    examples[group]=selected

class_table=pd.DataFrame(failure_rows).sort_values(['Group','Acceptance rate'],ascending=[True,False])
class_table.to_csv(RESULTS/'per_unknown_class_acceptance.csv',index=False)
print('Which unknown classes pass the frozen Vanilla MLS threshold?')
display(class_table)

fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
show_rows=[]
for r,(group,arr,values) in enumerate([('Near',vnear,vscore_near['MLS']),('Far',vfar,vscore_far['MLS'])]):
    for col,i in enumerate(examples[group]):
        fine=c100.classes[int(arr['labels'][i])]
        pred=classes[int(arr['logits'][i,:10].argmax())]
        score=float(values[i]); tau=thresholds['Vanilla MLS']
        axes[r,col].imshow(c100.data[int(arr['ids'][i])]); axes[r,col].axis('off')
        axes[r,col].set_title(f'{group}: {fine} → {pred}\nMLS {score:.2f} ≤ cutoff {tau:.2f}',fontsize=10)
        show_rows.append({'Group':group,'CIFAR-100 test id':int(arr['ids'][i]),'Unknown class':fine,
                          'Known prediction':pred,'MLS unknownness':score,'Threshold':tau,
                          'Interpretation':'plausible semantic overlap' if group=='Near' else 'surprising cross-category acceptance'})
fig.savefig(RESULTS/'six_accepted_unknowns.png',dpi=180,bbox_inches='tight')
plt.show()
examples_table=pd.DataFrame(show_rows)
examples_table.to_csv(RESULTS/'six_accepted_unknowns.csv',index=False)
display(examples_table.round(3))

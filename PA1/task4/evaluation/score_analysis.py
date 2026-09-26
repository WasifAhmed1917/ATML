# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 15
comp=pd.DataFrame(model_rows).set_index(['Model','Score'])
van=comp.loc[('Vanilla','MLS')]
for model in ['GCSC','PROSER']:
    target=comp.loc[(model,'MLS')]
    print(f'{model} − Vanilla MLS: CSA {100*(target["CSA"]-van["CSA"]):+.2f} points; '
          f'near AUROC {100*(target["AUROC near"]-van["AUROC near"]):+.2f} points; '
          f'far AUROC {100*(target["AUROC far"]-van["AUROC far"]):+.2f} points; '
          f'near reject {100*(target["Near reject"]-van["Near reject"]):+.2f} points; '
          f'far reject {100*(target["Far reject"]-van["Far reject"]):+.2f} points')
placeholder=comp.loc[('PROSER','placeholder')]
proser_mls=comp.loc[('PROSER','MLS')]
print(f'PROSER placeholder − PROSER MLS: near AUROC {100*(placeholder["AUROC near"]-proser_mls["AUROC near"]):+.2f} points; '
      f'far AUROC {100*(placeholder["AUROC far"]-proser_mls["AUROC far"]):+.2f} points')
rank_corr=pd.DataFrame({k:np.concatenate([vscore_near[k],vscore_far[k]]) for k in vscore_test}).corr(method='spearman')
rank_corr.to_csv(RESULTS/'vanilla_score_rank_correlation.csv')
print('Vanilla score rank agreement on the 1,600 fixed unknown test images:')
display(rank_corr.round(3))
from scipy.stats import rankdata
joined={s:np.concatenate([vscore_near[s],vscore_far[s]]) for s in vscore_test}
rank_gap=np.abs(rankdata(joined['MSP'])-rankdata(joined['MLS']))
disagree=[]
for i in np.argsort(rank_gap)[-6:][::-1]:
    group,local=('Near',int(i)) if i<len(vnear['ids']) else ('Far',int(i)-len(vnear['ids']))
    arr=vnear if group=='Near' else vfar
    disagree.append({'Group':group,'Unknown class':c100.classes[int(arr['labels'][local])],
                     'Known prediction':classes[int(arr['logits'][local,:10].argmax())],
                     'MSP':joined['MSP'][i],'MLS':joined['MLS'][i],
                     'Energy':joined['Energy'][i],'Mahalanobis':joined['Mahalanobis'][i],
                     'MSP versus MLS rank gap':rank_gap[i]})
disagreement_table=pd.DataFrame(disagree)
disagreement_table.to_csv(RESULTS/'score_rank_disagreements.csv',index=False)
print('Examples on which relative confidence (MSP) and logit magnitude (MLS) disagree most:')
display(disagreement_table.round(3))
for group in ('Near','Far'):
    print(group,'most often accepted classes and absorbing known labels:')
    display(class_table[class_table['Group']==group].head(4))
print('Interpret score differences using the frozen tables: MSP = relative class confidence; MLS = absolute top logit; '
      'Energy = evidence across logits; Mahalanobis = distance from training feature clusters. '
      'PROSER mixups cover boundaries between known classes, not every unseen direction. '
      'AUROC measures ranking; validation-calibrated rejection measures the specified operating point.')
(RESULTS/'environment.json').write_text(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,
    'python':platform.python_version(),'device':torch.cuda.get_device_name(0),'seed':SEED},indent=2))
print('Saved checkpoints, split and unknown IDs, frozen protocol, features/logits, tables, figures, and inspected cases to',ROOT)

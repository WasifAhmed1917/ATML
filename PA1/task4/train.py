# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 7
vanilla,vck=fit('Vanilla',100,0.1,vanilla_aug)
gcsc,gck=fit('GCSC',100,0.1,gcsc_aug)
proser,pck=fit('PROSER',50,1e-3,vanilla_aug,init_from=MODELS/'Vanilla_best.pt')
print('Selected validation accuracies:',{n:round(k['val_accuracy'],4) for n,k in [('Vanilla',vck),('GCSC',gck),('PROSER',pck)]})

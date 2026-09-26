# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 16
source_candidates=source_table[source_table.method.isin(['DAN 0.1','DAN 1','DAN 10'])]
source_choice=source_candidates.sort_values(['mean_source_macro_f1','method'],ascending=[False,True]).iloc[0]
choice={'criterion':'maximum mean macro-F1 across three source validations; lexicographic tie break',
        'selected_DAN_strength':source_choice['method'],
        'source_mean_macro_f1':float(source_choice['mean_source_macro_f1'])}
choice_file=OUT/'dan_strength_selected_without_target_labels.json'
if choice_file.exists():
    assert json.loads(choice_file.read_text())==choice,'Source-only decision changed after being saved.'
else:choice_file.write_text(json.dumps(choice,indent=2))
print('Source-validation choice (for analysis, main DAN remains λ=1):',choice)

task3_settings={'seed':SEED,'sources':DOMAINS,'unseen_target':TARGET,
    'shared_source_splits':str(split_file),'erm_checkpoint':str(OUT/'source_only_erm.pt'),
    'model':'ResNet18_Weights.IMAGENET1K_V1, same seven-class linear head',
    'preprocessing':'same 256 resize, 224 random train crop + flip, 224 center eval crop',
    'batch_per_source':8,'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,
    'epochs_max':30,'early_stop_patience':5,'checkpoint_selection':'mean source-validation macro-F1',
    'bn':'ImageNet running stats frozen; affine trainable',
    'dan_dg_main_lambda':1,'dan_dg_study_lambdas':[0.1,1,10],
    'dan_dg':'mean of three source-pair MMDs; same kernel as Task 2',
    'sam_main_rho':0.05,'sam_type':'non-adaptive, two passes, AdamW',
    'source_domain_probe':'balanced 70/30 multinomial logistic regression C=1 seed 6304',
    'sharpness_proxy':'fixed 32 per source, rho=0.05, eval mode',
    'hypothesis':'Increasing source alignment may lower source domain separability but over-alignment can damage class recognition.'}
lock_file=OUT/'task3_settings_locked_before_target_labels.json'
if lock_file.exists():
    assert json.loads(lock_file.read_text())==task3_settings,'Task 3 settings have changed after locking.'
else:lock_file.write_text(json.dumps(task3_settings,indent=2))
print('Task 3 settings locked before Task 2 target-label evaluation:',lock_file)

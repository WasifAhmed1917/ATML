# Notebook-derived stage: ATML_PA1_Task3.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 1
import hashlib, json, math, random, shutil, subprocess, sys, zipfile
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm.auto import tqdm
from IPython.display import display

SEED=6304
DOMAINS=('photo','art_painting','cartoon')
CLASSES=('dog','elephant','giraffe','guitar','horse','house','person')
STRENGTHS=(0.1,1.,10.)
TASK2=Path('/content/drive/MyDrive/ATML_PA1/Task2_PACS_stabilized/results')
OUT=Path('/content/drive/MyDrive/ATML_PA1/Task3_PACS/results')
OUT.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE=='cuda','Choose a Colab GPU runtime.'
def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_all(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
assert all((TASK2/p).is_file() for p in (
    'imagenet_initial.pt','source_only_erm.pt','source_only.pt',
    'splits/pacs_sketch_seed6304.json','task3_settings_locked_before_target_labels.json',
    'config.json','task2_summary.csv')),'Task 2 source checkpoint, split, config, or lock missing.'
lock=json.loads((TASK2/'task3_settings_locked_before_target_labels.json').read_text())
t2config=json.loads((TASK2/'config.json').read_text())
assert lock['seed']==SEED and tuple(lock['sources'])==DOMAINS and lock['unseen_target']=='sketch'
assert lock['dan_dg_main_lambda']==1 and lock['dan_dg_study_lambdas']==[0.1,1,10]
assert lock['sam_main_rho']==0.05 and lock['lr']==1e-4 and lock['weight_decay']==1e-4
assert t2config['global_grad_clip_norm']==1.0
assert hashlib.sha256((TASK2/'source_only.pt').read_bytes()).digest()==hashlib.sha256((TASK2/'source_only_erm.pt').read_bytes()).digest()
config={**lock,'results_dir':str(OUT),'erm_sha256':hashlib.sha256((TASK2/'source_only_erm.pt').read_bytes()).hexdigest(),
        'original_task3_lock':str(TASK2/'task3_settings_locked_before_target_labels.json'),
        'provenance_clarification':'Use shared global gradient clip 1.0 documented in the earlier Task 2 config; the original Task 3 lock is preserved.'}
config_path=OUT/'config.json'
if config_path.exists():
    assert json.loads(config_path.read_text())==config,'Task 3 configuration changed between runs.'
else:config_path.write_text(json.dumps(config,indent=2))
print('Device:',torch.cuda.get_device_name(0))
print('Original ERM checkpoint:',TASK2/'source_only_erm.pt')
print('Task 3 output:',OUT)

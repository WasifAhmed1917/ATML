# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 2
import json, math, random, shutil, zipfile, os
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from IPython.display import display
SEED=6304; DOMAINS=('photo','art_painting','cartoon'); TARGET='sketch'
CLASSES=('dog','elephant','giraffe','guitar','horse','house','person')
RUNS=(('Source-only',0.),('DAN 0.1',0.1),('DAN 1',1.),('DAN 10',10.),('DANN',1.),('CDAN',1.))
OUT=Path('/content/drive/MyDrive/ATML_PA1/Task2_PACS_stabilized/results'); OUT.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE=='cuda','Select a Colab GPU runtime.'
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_all(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
print('Device:',torch.cuda.get_device_name(0),'| PyTorch:',torch.__version__)
config={'seed':SEED,'sources':DOMAINS,'target':TARGET,'classes':CLASSES,'model':'torchvision resnet18 ResNet18_Weights.IMAGENET1K_V1',
 'batch_per_source':8,'batch_target':24,'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,
 'epochs_max':30,'early_stop_patience':5,'selection':'mean of three source-validation macro-F1',
 'bn':'ImageNet running statistics frozen, affine parameters trainable',
 'dan_lambdas':[0.1,1,10],'dann_cdan_grl':'2/(1+exp(-10p))-1, p=global update/max-30-epoch updates',
 'domain_loss_weight':1,'global_grad_clip_norm':1.0,
 'discriminator_feature_input':'L2-normalized feature times sqrt(512), for DANN and CDAN only',
 'discriminator_last_layer_init':'zeros','domain_discriminator':'256 ReLU dropout 0.5 2',
 'diagnostic':'equal source validation and target counts, 70/30 stratified, balanced logistic regression C=1'}
(OUT/'config.json').write_text(json.dumps(config,indent=2))

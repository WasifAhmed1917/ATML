# Notebook-derived stage: ATML_PA1_Task4.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 1
import os, json, random, copy, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from google.colab import drive

drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/ATML_PA1/Task4_CIFAR_OSR')
MODELS=ROOT/'checkpoints'; ARRAYS=ROOT/'arrays'; RESULTS=ROOT/'results'
for folder in (MODELS, ARRAYS, RESULTS): folder.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type=='cuda', 'Select a GPU runtime in Colab.'
SEED=6304

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
set_seed()
print('GPU:',torch.cuda.get_device_name(0),'| Torch:',torch.__version__,'| torchvision:',torchvision.__version__)

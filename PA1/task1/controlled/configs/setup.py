# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 3
import gc, json, random, shutil
import cv2
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from IPython.display import display, Markdown
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import STL10
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights
from torchvision.transforms import functional as TF, InterpolationMode
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import open_clip, torchvision, sklearn

SEED=6304
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=False
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE=='cuda', 'Select a Colab GPU runtime before continuing.'
from google.colab import drive
drive.mount('/content/drive')
OUT=Path('/content/drive/MyDrive/ATML_PA1/Task1_STL10_Controlled'); OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/content/stl10_data'); DATA.mkdir(parents=True,exist_ok=True)
BATCH=64
CLASSES=['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
NAMES=['ResNet-50','ViT-B/16','CLIP head','CLIP zero-shot']
PAIRS=[('airplane','ship'),('bird','cat'),('deer','horse'),('dog','monkey'),('car','truck')]
HUE_DEGREES=90
STYLE_WEIGHT=2e5
STYLE_STEPS=70
STYLE_LR=0.035
VIS_N=100
print(f'Device: {DEVICE} | torch {torch.__version__} | torchvision {torchvision.__version__} | open_clip {getattr(open_clip,"__version__","unknown")} | sklearn {sklearn.__version__}')
print('Output:', OUT)

# %% Original notebook cell 4
config=dict(seed=SEED,dataset='STL10',size=224,resize='PIL bicubic',split='stratified 80/20 official train',
    models={'ResNet-50':'ResNet50_Weights.IMAGENET1K_V2','ViT-B/16':'ViT_B_16_Weights.IMAGENET1K_V1','CLIP':'ViT-B-32 pretrained=openai'},
    head=dict(optimizer='AdamW',lr=1e-3,weight_decay=1e-4,max_epochs=50,patience=5,selection='validation accuracy'),
    prompt='a photo of a {class}.',hue_degrees=HUE_DEGREES,cue=dict(method='PyTorch Gatys / VGG19 style transfer',style_weight=STYLE_WEIGHT,steps=STYLE_STEPS,lr=STYLE_LR,content_weight=1,mask='GrabCut 2 iterations',texture_crop=96,texture_tile=112,init='masked content',pairs=PAIRS),
    translation=[0,8,16,32],patch_grid=[4,4],visualization=dict(method='t-SNE',seed=SEED,n=VIS_N,perplexity=30,metric='cosine',init='pca',learning_rate='auto'))
(OUT/'config.json').write_text(json.dumps(config,indent=2))

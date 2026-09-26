# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 7
RW = ResNet50_Weights.IMAGENET1K_V2
VW = ViT_B_16_Weights.IMAGENET1K_V1

from open_clip.pretrained import get_pretrained_cfg
clip_cfg = get_pretrained_cfg('ViT-B-32', 'openai')
assert 'mean' in clip_cfg and 'std' in clip_cfg

NORMALIZATIONS = {
    'ResNet-50': RW.transforms(),
    'ViT-B/16': VW.transforms(),
    'CLIP head': torchvision.transforms.Normalize(
        mean=clip_cfg['mean'], std=clip_cfg['std']
    ),
}
print({k: (v.mean, v.std) for k, v in NORMALIZATIONS.items()})

class ResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__(); self.net=resnet50(weights=RW); self.net.fc=nn.Identity()
    def forward(self,x): return self.net(x)
class ViTFeatures(nn.Module):
    def __init__(self): super().__init__(); self.net=vit_b_16(weights=VW)
    def forward(self,x):
        m=self.net; z=m._process_input(x)
        z=torch.cat((m.class_token.expand(z.shape[0],-1,-1),z),dim=1)
        return m.encoder(z)[:,0]
class CLIPFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.net, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', force_quick_gelu=True)
    def forward(self,x): return F.normalize(self.net.encode_image(x).float(),dim=-1)
def make_model(name):
    model={'ResNet-50':ResNetFeatures,'ViT-B/16':ViTFeatures,'CLIP head':CLIPFeatures}[name]().to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def features(model,name,images,batch=BATCH):
    norm=NORMALIZATIONS[name]
    mean=torch.tensor(norm.mean,device=DEVICE).view(1,3,1,1)
    std=torch.tensor(norm.std,device=DEVICE).view(1,3,1,1)
    result=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch):
            t=torch.from_numpy(np.ascontiguousarray(images[start:start+batch])).to(DEVICE)
            t=(t.permute(0,3,1,2).float()/255-mean)/std
            result.append(model(t).float().cpu().numpy())
    return np.concatenate(result)
assert len(Xtr)==4000 and len(Xva)==1000
print('Feature extraction is frozen; only the linear heads will train.')

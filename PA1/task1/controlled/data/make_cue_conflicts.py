# Notebook-derived stage: ATML_PA1_Task1_Controlled.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 15
from torchvision.models import vgg19, VGG19_Weights

style_net=vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:22].to(DEVICE).eval()
for parameter in style_net.parameters(): parameter.requires_grad_(False)
mean=torch.tensor([.485,.456,.406],device=DEVICE).view(1,3,1,1)
std=torch.tensor([.229,.224,.225],device=DEVICE).view(1,3,1,1)

def activations(rgb):
    z=(rgb-mean)/std; found={}
    for i,layer in enumerate(style_net):
        z=layer(z)
        if i in (0,5,10,19,21): found[i]=z.clone()
    return found

def gram(t):
    _,channels,height,width=t.shape
    f=t.reshape(channels,height*width)
    return (f@f.T)/(channels*height*width)

def foreground(image):

    mask=np.zeros((224,224),np.uint8)
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    cv2.grabCut(image,mask,(12,12,200,200),bg,fg,2,cv2.GC_INIT_WITH_RECT)
    fg_mask=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    fg_mask=cv2.morphologyEx(fg_mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return fg_mask

def texture_tile(image):

    crop=Image.fromarray(image[64:160,64:160]).resize((112,112),Image.Resampling.BICUBIC)
    patch=np.asarray(crop)
    return np.concatenate([np.concatenate([patch,patch[:,::-1]],axis=1),
                           np.concatenate([patch[::-1],patch[::-1,::-1]],axis=1)],axis=0)

def stylize(content,style):
    mask=foreground(content)
    m=torch.from_numpy((mask/255).copy()).to(DEVICE,dtype=torch.float32)[None,None]
    foreground_rgb=(content.astype(np.float32)/255)*m.cpu().numpy()[0,0,:, :,None]

    masked=np.clip(foreground_rgb+(1-m.cpu().numpy()[0,0,:,:,None]),0,1)
    c=torch.from_numpy(masked).to(DEVICE).permute(2,0,1)[None].float()
    tiled=texture_tile(style)
    t=torch.from_numpy(np.ascontiguousarray(tiled)).to(DEVICE).permute(2,0,1)[None].float()/255
    with torch.no_grad():
        content_target=activations(c)[21].detach()
        style_targets={i:gram(a).detach() for i,a in activations(t).items() if i!=21}
    pixels=c.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([pixels],lr=STYLE_LR)
    for _ in range(STYLE_STEPS):
        opt.zero_grad(set_to_none=True)
        z=pixels*m+(1-m)
        out=activations(z)
        content_loss=F.mse_loss(out[21],content_target)
        style_loss=sum(F.mse_loss(gram(out[i]),style_targets[i]) for i in style_targets)
        loss=content_loss+STYLE_WEIGHT*style_loss
        loss.backward();opt.step()
        with torch.no_grad(): pixels.clamp_(0,1)
    final=(pixels.detach()*m+1-m).clamp(0,1)
    rgb=(final[0].permute(1,2,0)*255).round().byte().cpu().numpy()
    del pixels,opt,c,t,final,out
    return rgb,mask,tiled
print('Iterative VGG19 transfer ready; backbones and heads remain unchanged.')

# %% Original notebook cell 16
review_file=OUT/'cue_review.csv'; parts=OUT/'cue_parts';parts.mkdir(exist_ok=True)
rng=np.random.default_rng(SEED)
planned=[]
for a,b in PAIRS:
    for source,target in [(a,b),(b,a)]:
        sources=np.flatnonzero(y==CLASSES.index(source))
        donors=np.flatnonzero(y==CLASSES.index(target))
        assert len(sources)>=40 and len(donors)>=2
        for ci in rng.choice(sources,40,replace=False):
            si=int(rng.choice(donors))
            planned.append(dict(candidate_id=len(planned),content_id=int(ci),style_id=si,
                                content_class=source,style_class=target))
assert len(planned)==400
if not review_file.exists(): pd.DataFrame(planned).assign(accept='').to_csv(review_file,index=False)
else:
    prior=pd.read_csv(review_file,keep_default_na=False)
    assert prior.iloc[:400][list(planned[0])].to_dict('records')==planned, 'Existing review CSV has different sampling metadata.'

for k in tqdm(range(10),desc='Transfer directions'):
    part=parts/f'{k:02d}.npz'
    if part.exists(): continue
    batch=[];masks=[]
    for item in tqdm(planned[k*40:(k+1)*40],desc=f'{planned[k*40]["content_class"]} → {planned[k*40]["style_class"]}',leave=False):
        result,mask,_=stylize(X[item['content_id']],X[item['style_id']])
        batch.append(result);masks.append(mask)
    np.savez_compressed(part,images=np.stack(batch),masks=np.stack(masks))
CAND=np.concatenate([np.load(parts/f'{i:02d}.npz')['images'] for i in range(10)])
MASKS=np.concatenate([np.load(parts/f'{i:02d}.npz')['masks'] for i in range(10)])
for addition in sorted(parts.glob('expanded_*.npz'),key=lambda p:int(p.stem.split('_')[1])):
    with np.load(addition) as d:
        CAND=np.concatenate([CAND,d['images']]);MASKS=np.concatenate([MASKS,d['masks']])
assert CAND.shape[1:]==(224,224,3) and MASKS.shape[1:]==(224,224)
review=pd.read_csv(review_file,dtype={'accept':'string'},keep_default_na=False)
assert len(review)==len(CAND) and review.candidate_id.tolist()==list(range(len(CAND)))
print('Candidates:',len(CAND),'| initial 40 per direction | Review:',review_file)

# %% Original notebook cell 17
SHEET=OUT/'cue_contact_sheets';SHEET.mkdir(exist_ok=True)
for start in range(0,len(CAND),10):
    path=SHEET/f'{start:03d}-{min(start+9,len(CAND)-1):03d}.jpg'
    if path.exists(): continue
    canvas=Image.new('RGB',(4*136,10*140),'white'); draw=ImageDraw.Draw(canvas)
    for row,k in enumerate(range(start,min(start+10,len(CAND)))):
        item=review.iloc[k]
        content=X[int(item.content_id)]
        isolated=(content.astype(np.float32)*(MASKS[k,:, :,None]/255)+255*(1-MASKS[k,:, :,None]/255)).round().astype(np.uint8)
        panels=[content,isolated,texture_tile(X[int(item.style_id)]),CAND[k]]
        for j,img in enumerate(panels): canvas.paste(Image.fromarray(img).resize((128,128)),(j*136,row*140))
        draw.text((2,row*140+128),f'{k}: {item.content_class} / {item.style_class}',fill='black')
    canvas.save(path,quality=90)
print('Saved 40 contact sheets:',SHEET)
display(Image.open(SHEET/'000-009.jpg'))

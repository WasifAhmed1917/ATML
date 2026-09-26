# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 14
REPO=Path('/content/pytorch-AdaIN')
if not REPO.exists(): subprocess.run(['git','clone','--quiet','https://github.com/naoto0804/pytorch-AdaIN.git',str(REPO)],check=True)
commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
config['adain_source']='https://github.com/naoto0804/pytorch-AdaIN'; config['adain_commit']=commit
(OUT/'config.json').write_text(json.dumps(config,indent=2))
import urllib.request
weight_paths={}
for key in ['vgg_normalised','decoder']:
    path=OUT/(key+'.pth'); url=f'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/{key}.pth'
    if not path.exists(): urllib.request.urlretrieve(url,path)
    assert path.stat().st_size>1000000, f'Incomplete AdaIN download: {path}'
    weight_paths[key]=path
sys.path.insert(0,str(REPO))
import net
from function import adaptive_instance_normalization as adain
vgg=net.vgg.to(DEVICE).eval(); dec=net.decoder.to(DEVICE).eval()
vgg.load_state_dict(torch.load(weight_paths['vgg_normalised'],map_location=DEVICE,weights_only=True))
dec.load_state_dict(torch.load(weight_paths['decoder'],map_location=DEVICE,weights_only=True))
vgg=nn.Sequential(*list(vgg.children())[:31]).eval()
for module in (vgg,dec):
    for p in module.parameters(): p.requires_grad_(False)
print('AdaIN commit:',commit)

# %% Original notebook cell 15
candidate_file=OUT/'cue_candidates.npz'; review_file=OUT/'cue_review.csv'
if not candidate_file.exists():
    rng=np.random.default_rng(SEED)
    candidates=[]; info=[]
    def adain_image(content,style):
        inp=torch.from_numpy(np.ascontiguousarray(content)).to(DEVICE).permute(2,0,1)[None].float()/255
        sty=torch.from_numpy(np.ascontiguousarray(style)).to(DEVICE).permute(2,0,1)[None].float()/255
        with torch.inference_mode():
            cf=vgg(inp); sf=vgg(sty)
            mixed=ADA_ALPHA*adain(cf,sf)+(1-ADA_ALPHA)*cf
            out=dec(mixed).clamp(0,1)[0].permute(1,2,0)
        return (out.mul(255).round().byte().cpu().numpy())
    for ca,cb in PAIRS:
        for content_class,style_class in [(ca,cb),(cb,ca)]:
            pool=np.flatnonzero(y==CLASSES.index(content_class))
            styles=np.flatnonzero(y==CLASSES.index(style_class))
            assert len(pool)>=40 and len(styles)>=1, 'Not enough selected STL-10 examples for candidate generation.'
            for content_id in tqdm(rng.choice(pool,40,replace=False),desc=f'{content_class} → {style_class}'):
                style_id=int(rng.choice(styles)); candidate_id=len(candidates)
                candidates.append(adain_image(X[content_id],X[style_id]))
                info.append(dict(candidate_id=candidate_id,content_id=int(content_id),style_id=style_id,
                    content_class=content_class,style_class=style_class))
    np.savez_compressed(candidate_file,images=np.stack(candidates))
    pd.DataFrame(info).assign(accept='').to_csv(review_file,index=False)
assert candidate_file.exists() and review_file.exists()
CAND=np.load(candidate_file)['images']; review=pd.read_csv(review_file,dtype={'accept':'string'},keep_default_na=False)
assert len(CAND)==len(review)==400
print('Generated candidates:',len(CAND),'| review:',review_file)

# %% Original notebook cell 16
SHEET=OUT/'cue_contact_sheets'; SHEET.mkdir(exist_ok=True)
for start in range(0,len(CAND),10):
    canvas=Image.new('RGB',(3*136,10*140),'white'); d=ImageDraw.Draw(canvas)
    for row,k in enumerate(range(start,min(start+10,len(CAND)))):
        m=review.iloc[k]
        for j,img in enumerate([X[int(m.content_id)],X[int(m.style_id)],CAND[k]]):
            canvas.paste(Image.fromarray(img).resize((128,128)),(j*136,row*140))
        d.text((2,row*140+128),f'{k}: {m.content_class} / {m.style_class}',fill='black')
    path=SHEET/f'{start:03d}-{min(start+9,len(CAND)-1):03d}.jpg'; canvas.save(path,quality=85)
print('Contact sheets:',SHEET,'(40 pages, 10 triplets each).')
display(Image.open(SHEET/'000-009.jpg').resize((408,1400)))

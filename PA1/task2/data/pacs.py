# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 4
PACS_INPUT=Path('/content/PACS')
HF_REPO='Azeez577/PACS'
HF_REVISION='46a0a83'  # PACS.zip upload commit
HF_SHA256='42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147'
if not PACS_INPUT.exists():
    import hashlib, subprocess, sys
    archive=Path('/content/PACS.zip')
    if archive.exists() and not zipfile.is_zipfile(archive): archive.unlink()
    if not archive.exists():
        subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub'],check=True)
        from huggingface_hub import hf_hub_download
        archive=Path(hf_hub_download(repo_id=HF_REPO,repo_type='dataset',
                                     filename='PACS.zip',revision=HF_REVISION,
                                     local_dir='/content'))
    assert zipfile.is_zipfile(archive),'The downloaded PACS file is not a valid ZIP.'
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==HF_SHA256, (
        'PACS archive hash mismatch. Delete /content/PACS.zip and retry the download.')
    PACS_INPUT=archive
if PACS_INPUT.is_file():
    assert PACS_INPUT.suffix.lower()=='.zip','Expected a folder or .zip.'
    extract=Path('/content/PACS_extracted'); extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(PACS_INPUT) as z:
        assert z.testzip() is None,'PACS archive has a damaged member; delete /content/PACS.zip and retry.'
        for info in z.infolist():
            path=(extract/info.filename).resolve()
            assert path.is_relative_to(extract.resolve()),'Unsafe archive path'
        z.extractall(extract)
    search_root=extract
else: search_root=PACS_INPUT
candidates=[p for p in [search_root,*search_root.rglob('*')] if p.is_dir() and
            all((p/d).is_dir() for d in (*DOMAINS,TARGET))]
assert len(candidates)==1, f'Expected exactly one PACS domain root, found {candidates[:4]}'
ROOT=candidates[0]; IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
all_paths={}
for domain in (*DOMAINS,TARGET):
    all_paths[domain]=sorted(p for p in (ROOT/domain).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if domain in DOMAINS:
        found={p.parent.name.lower() for p in all_paths[domain]}
        assert found==set(CLASSES),(domain,found)
    assert len(all_paths[domain])>100,(domain,len(all_paths[domain]))
assert sum(len(paths) for paths in all_paths.values())==9991,'Expected the complete 9,991-image PACS dataset.'
config['dataset_source']=f'https://huggingface.co/datasets/{HF_REPO}/blob/{HF_REVISION}/PACS.zip'
config['dataset_sha256']=HF_SHA256
config['dataset_domain_counts']={d:len(paths) for d,paths in all_paths.items()}
(OUT/'config.json').write_text(json.dumps(config,indent=2))
print('PACS root:',ROOT)
print('Domain sizes:',{d:len(v) for d,v in all_paths.items()})

split_file=OUT/'splits'/'pacs_sketch_seed6304.json'; split_file.parent.mkdir(exist_ok=True)
if split_file.exists():
    splits=json.loads(split_file.read_text())
else:
    splits={}
    for domain in DOMAINS:
        paths=all_paths[domain]
        labels=[CLASSES.index(p.parent.name.lower()) for p in paths]
        tr,va=train_test_split(np.arange(len(paths)),test_size=0.2,stratify=labels,random_state=SEED)
        splits[domain]={'train':[str(paths[i].relative_to(ROOT)) for i in tr],
                        'val':[str(paths[i].relative_to(ROOT)) for i in va]}
    split_file.write_text(json.dumps(splits,indent=2))
for domain in DOMAINS:
    expected={str(p.relative_to(ROOT)) for p in all_paths[domain]}
    tr=set(splits[domain]['train']); va=set(splits[domain]['val'])
    assert not tr&va and tr|va==expected,(domain,'Split/data mismatch')
    print(domain,'train',len(tr),'val',len(va))
target_rel=[str(p.relative_to(ROOT)) for p in all_paths[TARGET]]

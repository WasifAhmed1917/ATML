from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def make_audit(cfg):
    import json
    from task4_safety.runtime import POLICIES,output_dir,validate_protocol,load_records,write_json,object_hash
    validate_protocol(cfg);out=output_dir(cfg)
    base=load_records(out/'generated_sft.jsonl')
    ids=fixed_audit_ids(base,int(cfg['manual_audit_per_class']),int(cfg['seed']))
    identity={'xstest_ids':ids,'rule':'released numpy default_rng(seed), 30 SAFE then 30 UNSAFE; sorted IDs',
              'seed':int(cfg['seed']),'n_prompt_ids':len(ids),'n_response_labels':len(ids)*len(POLICIES)}
    manifest=out/'manual_audit_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Existing fixed audit IDs differ')
    write_json(manifest,identity)
    selected=[]
    for policy in POLICIES:
        rows=load_records(out/f'generated_{policy}.jsonl');by_id={r['xstest_id']:r for r in rows}
        for xid in ids:
            r=by_id[xid]
            selected.append({k:r[k] for k in ('xstest_id','policy','benchmark_class','type','prompt','response')})
    # Neither AI labels/confidence nor rationale are present. Keep existing human work intact.
    frame=pd.DataFrame(selected);frame['manual_label']='';frame['manual_notes']=''
    sheet=out/'manual_audit_blind.csv'
    if sheet.exists():
        old=pd.read_csv(sheet,keep_default_na=False)
        if object_hash(old[frame.columns[:-2]].to_dict('records'))!=object_hash(frame[frame.columns[:-2]].to_dict('records')):
            raise ValueError('Existing manual sheet responses/IDs differ')
    else:frame.to_csv(sheet,index=False)
    pd.DataFrame({'xstest_id':ids}).to_csv(out/'manual_audit_ids.csv',index=False)
    print(f'Blind audit: {len(ids)} fixed prompt IDs x {len(POLICIES)} policies = {len(frame)} response labels. Fill manual_label before viewing AI labels.',flush=True)
    return frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');a=p.parse_args();make_audit(load_yaml(a.config))
if __name__=='__main__':main()

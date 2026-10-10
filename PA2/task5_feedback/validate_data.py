"""Preflight the fixed release, all diagnostic variants and supplied adapters."""
import argparse
from common.data import load_yaml, repo_path
from task5_feedback.runtime import (POLICIES, DATA, load_dataset, row_id, file_hash, output_dir,
                                   adapter_identity, validate_protocol, write_json, provenance)
from task5_feedback.score_perturbations import load_diagnostic_groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml')
    p.add_argument('--data-only',action='store_true');a=p.parse_args()
    cfg=load_yaml(a.config);validate_protocol(cfg);manifest={**provenance(cfg),'datasets':{},'adapters':{}}
    for domain,(key,_,_) in DATA.items():
        rows=load_dataset(cfg,domain)
        manifest['datasets'][domain]={'n_rows':len(rows),'sha256':file_hash(repo_path(cfg['paths'][key])),
                                     'record_ids':[row_id(r,domain) for r in rows]}
        print(domain,len(rows),'fixed rows verified',flush=True)
    load_diagnostic_groups(cfg['paths']['task5_diagnostics'])
    if not a.data_only:
        for name in POLICIES:
            path,sha=adapter_identity(cfg,name);manifest['adapters'][name]={'path':str(path) if path else None,'sha256':sha}
            print(name,'original SFT' if path is None else path,flush=True)
    write_json(output_dir(cfg)/'fixed_inputs.json',manifest)
    print('Task 5 release preflight passed.',flush=True)

if __name__=='__main__':main()

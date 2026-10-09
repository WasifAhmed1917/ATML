"""Read-only preflight: exact release protocol and completed standard checkpoints."""
import argparse
import json
from common.data import load_yaml
from task4_safety.generate_responses import load_xstest
from task4_safety.make_audit_sheet import fixed_audit_ids
from task4_safety.runtime import POLICIES,validate_protocol,validate_adapter,file_hash,output_dir,write_json


def validate(config='configs/feedback.yaml'):
    cfg=load_yaml(config);validate_protocol(cfg);frame=load_xstest(cfg)
    if len(frame)!=450 or frame.benchmark_class.value_counts().to_dict()!= {'SAFE':250,'UNSAFE':200}:
        raise ValueError('Expected full released XSTest: 450 prompts (250 SAFE, 200 UNSAFE)')
    ids=fixed_audit_ids(frame.to_dict('records'),30,int(cfg['seed']))
    policies={}
    for name in POLICIES:
        _,sha=validate_adapter(cfg,name);policies[name]=sha
    result={'n_prompts':len(frame),'class_counts':frame.benchmark_class.value_counts().to_dict(),
            'n_categories':int(frame.type.nunique()),'audit_prompt_ids':ids,'n_audit_prompt_ids':60,
            'n_audit_responses':240,'policy_adapter_sha256':policies,
            'xstest_sha256':file_hash(__import__('common.data',fromlist=['repo_path']).repo_path(cfg['paths']['xstest']))}
    write_json(output_dir(cfg)/'data_manifest.json',result);print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');a=p.parse_args();validate(a.config)
if __name__=='__main__':main()

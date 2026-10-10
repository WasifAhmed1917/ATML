"""Clean-vs-variant controlled pairs, including a clean self-pair identity control."""
from __future__ import annotations
import argparse
from collections import defaultdict
from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.metrics import diagnostic_rates
from task5_feedback.judge_pairs import score_pairs
from task5_feedback.runtime import validate_protocol,load_dataset,output_dir,write_json,object_hash

EXPECTED_VARIANTS = {'clean_correct','corrupt_reasoning_correct_final','good_reasoning_wrong_final',
                     'persuasive_filler_correct','gold_distractor_wrong_final'}
VARIANTS = ('clean_correct','corrupt_reasoning_correct_final','good_reasoning_wrong_final',
            'persuasive_filler_correct','gold_distractor_wrong_final')


def load_diagnostic_groups(path):
    rows=read_jsonl(path); by_problem=defaultdict(dict)
    for row in rows:
        pid=str(row['problem_id']); variant=row['variant_type']
        if variant not in EXPECTED_VARIANTS or variant in by_problem[pid]:
            raise ValueError('Unknown or duplicate diagnostic variant')
        by_problem[pid][variant]=row
    for pid,variants in by_problem.items():
        if set(variants)!=EXPECTED_VARIANTS:raise ValueError(f'Problem {pid} missing variants')
        if len({str(r['gold_final']) for r in variants.values()})!=1 or len({r['question'] for r in variants.values()})!=1:
            raise ValueError('Diagnostic problem identity differs across variants')
        if any(r.get('manual_validation') is not True for r in variants.values()):
            raise ValueError('Use staff-validated diagnostics unchanged')
        for row in variants.values():
            if exact_reward(row['response'],str(row['gold_final']))!=row['expected_exact_reward']:
                raise ValueError('Diagnostic response disagrees with course verifier expectation')
    if len(by_problem)!=20:raise ValueError('Expected all 20 diagnostic problems')
    return by_problem


def controlled_pairs(groups):
    pairs=[]
    for pid,variants in groups.items():
        clean=variants['clean_correct']
        for variant in VARIANTS:
            b=variants[variant]
            pairs.append({'problem_id':pid,'variant_type':variant,'question':clean['question'],
                          'gold_final':str(clean['gold_final']),'a_response':clean['response'],'b_response':b['response'],
                          'a_exact_reward':exact_reward(clean['response'],str(clean['gold_final'])),
                          'b_exact_reward':exact_reward(b['response'],str(clean['gold_final']))})
    return pairs


def summarize_diagnostics(records):
    summary={'n_problems':len({r['problem_id'] for r in records}),'n_responses':len(records),
             'n_controlled_pairs':len(records),'comparison':'A=clean, B=variant; clean category is self-pair',
             'categories':{},'sensitivity':{}}
    for mechanism in ('verifier','judge'):
        summary['categories'][mechanism]={v:diagnostic_rates([r for r in records if r['variant_type']==v],mechanism) for v in VARIANTS}
        cats=summary['categories'][mechanism]
        summary['sensitivity'][mechanism]={
            'S_reason':cats['corrupt_reasoning_correct_final']['better_response_rate'],
            'S_outcome':cats['good_reasoning_wrong_final']['better_response_rate'],
            'reasoning_tie_rate':cats['corrupt_reasoning_correct_final']['tie_rate'],
            'reasoning_wrong_rate':cats['corrupt_reasoning_correct_final']['wrong_preference_rate'],
            'outcome_tie_rate':cats['good_reasoning_wrong_final']['tie_rate'],
            'outcome_wrong_rate':cats['good_reasoning_wrong_final']['wrong_preference_rate'],
            'filler_preference_rate':cats['persuasive_filler_correct']['variant_preference_rate'],
            'distractor_robustness':cats['gold_distractor_wrong_final']['better_response_rate']}
    return summary


def run(config='configs/feedback.yaml',resume=False):
    cfg=load_yaml(config);validate_protocol(cfg);rows=load_dataset(cfg,'diagnostics')
    groups=load_diagnostic_groups(cfg['paths']['task5_diagnostics'])
    inputs=controlled_pairs(groups);out=output_dir(cfg)
    records=score_pairs(cfg,inputs,out/'diagnostic_pairs.jsonl',resume,'diagnostic_pairwise',smoke=False,
                        diagnostic_sha256=object_hash(rows),judge_new_tokens=4,judge_do_sample=False,
                        orientation='released hash-balanced A/B',unparseable_rule='released TIE fallback')
    summary=summarize_diagnostics(records);write_json(out/'diagnostic_summary.json',summary)
    print(summary['sensitivity'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.resume)

if __name__=='__main__':main()

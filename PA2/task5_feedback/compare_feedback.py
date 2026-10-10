"""Export all PA Task 5 evidence, with explicit tie and failure conventions."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from common.data import load_yaml
from task5_feedback.runtime import (POLICIES,DOMAINS,DATA,output_dir,require_complete,write_json,
                                   validate_protocol,load_dataset,object_hash,provenance)
from task5_feedback.evaluate_math import aggregate
from task5_feedback.score_perturbations import VARIANTS,summarize_diagnostics,controlled_pairs,load_diagnostic_groups


def write_csv(path,rows):
    if not rows:raise ValueError('Empty evidence table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def validate_saved(records,inputs):
    if len(records)!=len(inputs):raise ValueError('Incomplete evidence')
    for i,(r,input_row) in enumerate(zip(records,inputs)):
        if r['row_index']!=i or r['input_sha256']!=object_hash(input_row):
            raise ValueError('Evidence does not match fixed input identity/order')


def export(config='configs/feedback.yaml'):
    cfg=load_yaml(config);validate_protocol(cfg);out=output_dir(cfg)
    summaries={};policy_rows=[];failures=[];all_pairs=[];generations={}
    for domain in DOMAINS:
        rows=load_dataset(cfg,domain);summaries[domain]=aggregate(cfg,domain)
        for policy in POLICIES:
            path=out/f'{domain}_{policy}_responses.jsonl'
            responses,meta=require_complete(path,path.with_suffix('.meta.json'),len(rows));validate_saved(responses,rows)
            generations[(domain,policy)]=responses
            s=summaries[domain]['policies'][policy]
            policy_rows.append({'dataset':domain,'policy':policy,**{k:v for k,v in s.items() if not isinstance(v,dict)}})
            for failure in ('correct_final','no_designated_final','incorrect_designated_final'):
                n=s['failure_counts'].get(failure,0)
                failures.append({'dataset':domain,'policy':policy,'failure_type':failure,'count':n,'fraction':n/len(rows)})
            if policy!='sft':
                path=out/f'{domain}_{policy}_pairs.jsonl'
                pairs,pmeta=require_complete(path,path.with_suffix('.meta.json'),len(rows))
                if pmeta['config_sha256']!=object_hash(cfg):raise ValueError('Pair config differs')
                baseline=generations[(domain,'sft')]
                expected=[{'prompt_id':a['prompt_id'],'dataset':domain,'policy':policy,'question':a['question'],
                           'a_response':a['response'],'b_response':b['response'],
                           'a_exact_reward':a['exact_reward'],'b_exact_reward':b['exact_reward']} for a,b in zip(responses,baseline)]
                validate_saved(pairs,expected);all_pairs.extend(pairs)
    diag_path=out/'diagnostic_pairs.jsonl'
    diagnostics,meta=require_complete(diag_path,diag_path.with_suffix('.meta.json'),100)
    if meta['config_sha256']!=object_hash(cfg):raise ValueError('Diagnostic config differs')
    expected=controlled_pairs(load_diagnostic_groups(cfg['paths']['task5_diagnostics']))
    validate_saved(diagnostics,expected)
    diag=summarize_diagnostics(diagnostics);write_json(out/'diagnostic_summary.json',diag)
    diag_rows=[{'mechanism':m,'variant_type':v,**s} for m,cats in diag['categories'].items() for v,s in cats.items()]
    write_csv(out/'summary.csv',policy_rows);write_csv(out/'failure_types.csv',failures)
    write_csv(out/'diagnostic_rates.csv',diag_rows)
    write_csv(out/'sensitivity.csv',[{'mechanism':m,**s} for m,s in diag['sensitivity'].items()])
    drops=[]
    for policy in POLICIES:
        a=summaries['gsm']['policies'][policy];b=summaries['transfer']['policies'][policy]
        drops.append({'policy':policy,'gsm_accuracy':a['exact_accuracy'],'svamp_accuracy':b['exact_accuracy'],
                      'accuracy_drop_pp':100*(a['exact_accuracy']-b['exact_accuracy']),
                      'format_compliance_drop_pp':100*(a['format_compliance']-b['format_compliance']),
                      'response_length_change_tokens':b['response_length_mean']-a['response_length_mean'],
                      'ai_win_rate_drop_pp':100*(a['ai_win_rate_vs_sft']-b['ai_win_rate_vs_sft']) if policy!='sft' else None})
    write_csv(out/'transfer_drops.csv',drops)
    agreements=[]
    for d in DOMAINS:
        for p in ('rlvr','rlaif'):
            s=summaries[d]['policies'][p]
            for key,value in s['contingency'].items():
                verifier,judge=key.split('/');agreements.append({'dataset':d,'policy':p,'verifier':verifier,'judge':judge,'count':value})
    write_csv(out/'agreement_contingency.csv',agreements)
    # Record disagreement cases; they require interpretation, rather than an
    # automatic assertion that the judge or the response reasoning is wrong.
    mismatch=[r for r in all_pairs if r['verifier_preference']!=r['judge_preference']]
    write_json(out/'policy_disagreements.json',mismatch)
    examples=[]
    for variant in VARIANTS[1:]:
        candidates=[r for r in diagnostics if r['variant_type']==variant]
        disagreement=[r for r in candidates if r['judge_preference']!=r['verifier_preference']]
        chosen=(disagreement or candidates)[0]
        examples.append({**chosen,'selection':'first fixed-order disagreement' if disagreement else 'first fixed-order pair'})
    write_json(out/'qualitative_diagnostics.json',examples)
    write_csv(out/'qualitative_diagnostics.csv',[{k:r[k] for k in ('problem_id','variant_type','question','gold_final',
        'a_response','b_response','a_exact_reward','b_exact_reward','verifier_preference','judge_preference','selection')} for r in examples])
    cost={'generated_responses':sum(len(v) for v in generations.values()),
          'generated_tokens':sum(r['response_tokens'] for rs in generations.values() for r in rs),
          'generation_seconds':sum(r['generation_seconds_share'] for rs in generations.values() for r in rs),
          'judge_pairs':len(all_pairs)+len(diagnostics),
          'judge_cache_hits':sum(r['judge_cache_hit'] for r in all_pairs+diagnostics),
          'judge_seconds_recorded':sum(r['judge_seconds'] for r in all_pairs+diagnostics),
          'verifier_calls_in_reported_pairs':2*(len(all_pairs)+len(diagnostics)),
          'scope':'Recorded evaluation cost only; generation time is amortized over committed responses and excludes loading/replay overhead. Cached judge time does not reconstruct prior interrupted inference.'}
    write_json(out/'inference_cost.json',cost)
    write_json(out/'comparison.json',{**provenance(cfg),'in_domain':summaries['gsm'],'transfer':summaries['transfer'],
                                     'diagnostics':diag,'transfer_drops':drops,'inference_cost':cost})
    write_json(out/'completion_checks.json',{'status':'complete','full_fixed_sets':True,'course_supplied_adapters':True,
              'gsm_prompts_per_policy':300,'svamp_prompts_per_policy':100,'diagnostic_problems':20,
              'diagnostic_categories':5,'diagnostic_pairs':100,'responses':1200,'policy_pairs':800,
              'no_new_training':True,'smoke_results_excluded':True})
    notes=["# Task 5 evidence conventions", "",
           "All results use the supplied SFT base, RLVR and RLAIF policies, without new training. Policy A is the evaluated adapter; B is SFT. Win rate excludes ties; the separate tie-adjusted rate assigns ties half a win. SFT has no fabricated self-judge score.","",
           "Verifier–judge agreement is three-way A/B/TIE agreement. Also inspect strict-verifier agreement and the judge's distinctions among verifier ties. A distinction between equally correct finals can reflect useful reasoning feedback or style bias; this metric alone cannot identify which.","",
           "The clean diagnostic category is an identity self-pair, expected to tie. Its better-response rate is undefined (blank), and non-ties are incorrect distinctions. For the other four categories, A is clean. Verifier ties on reasoning/filler are expected invariance, not an accuracy error. Filler preference reports how often B beats the clean response.","",
           "S_reason is clean versus corrupted reasoning with the final held correct. S_outcome is clean versus good reasoning/wrong final only; the gold-distractor category is reported separately. Inspect qualitative_diagnostics.csv alongside sensitivity.csv before attributing sensitivity to reasoning quality.","",
           "The unchanged course judge generates four tokens greedily, balances A/B orientation by its content hash, and parses unmatched outputs as TIE. feedback.yaml's generic judge_max_new_tokens=64 belongs to the safety judge; it does not override Task 5's supplied four-token pairwise judge. Ties can therefore include parser fallback.","",
           "Exact reward uses the released last designated #### numeric field, never a gold number appearing only in reasoning. Format compliance requires one designated field on the last line; correctness and formatting are reported separately. Failure types are observable final-answer failures, not automatically diagnosed reasoning errors. Truncation flags identify generation-cap effects.","",
           "Transfer drops are GSM8K minus SVAMP, in percentage points. Negative drop is improvement. The two benchmarks differ in difficulty; a drop is not causal evidence about general reasoning ability.","",
           "Feedback discussion: exact verification has broad objective coverage only where reliable answer keys exist, negligible inference cost, and no feedback on reasoning quality when the final is correct. AI feedback ranks reasoning/relevance beyond the final but inherits this judge's errors and style bias and requires model inference. Compare diagnostic ranking patterns and costs, not binary rewards and pairwise win rates as calibrated utilities.","",
           "Use the saved qualitative pairs to discuss reasoning-only corruption, irrelevant persuasive filler, and wrong-final/gold-distractor behavior. No Task 4 human labels are required here. Do not use these evaluations to tune earlier policies."]
    (out/'evidence_notes.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')
    print('Full Task 5 evidence exported:',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/feedback.yaml');a=p.parse_args();export(a.config)

if __name__=='__main__':main()

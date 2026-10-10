"""Behavioral checks of verifier sensitivity, pair orientation, and recovery."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from common.data import load_yaml
from task5_feedback.rlvr import exact_reward,extract_designated_final
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.metrics import response_metrics,verifier_preference,agreement,diagnostic_rates
from task5_feedback.runtime import (load_records,append_record,write_json,object_hash,resume_records,
                                   load_dataset,validate_protocol,output_dir)
from task5_feedback.score_perturbations import controlled_pairs,load_diagnostic_groups,summarize_diagnostics
from task5_feedback.judge_pairs import restore_cache,score_pairs


class FeedbackTests(unittest.TestCase):
    def test_final_field_not_gold_distractor(self):
        self.assertEqual(exact_reward('Maybe 42, discarded. Final:\n#### 7','42'),0.)
        self.assertEqual(exact_reward('42 is in the reasoning only','42'),0.)
        self.assertEqual(exact_reward('#### 42\nCorrection: #### 7','42'),0.)
        self.assertEqual(exact_reward('#### 1,234.0','1234'),1.)
        self.assertEqual(extract_designated_final('#### -7.5'),'-7.5')

    def test_correctness_and_format_are_separate(self):
        r=response_metrics('#### 7\nMore irrelevant filler','7')
        self.assertEqual(r['exact_reward'],1.)
        self.assertFalse(r['format_compliant'])
        self.assertTrue(response_metrics('Steps.\n#### 7','7')['format_compliant'])
        self.assertEqual(response_metrics('No final','7',True)['failure_type'],'no_designated_final')
        self.assertTrue(response_metrics('No final','7',True)['truncation_failure'])

    def test_ties_have_explicit_denominators(self):
        pairs=[{'verifier_preference':'TIE','judge_preference':'A'},
               {'verifier_preference':'A','judge_preference':'A'},
               {'verifier_preference':'B','judge_preference':'TIE'}]
        a=agreement(pairs)
        self.assertAlmostEqual(a['three_way_agreement'],1/3)
        self.assertEqual(a['judge_agreement_given_verifier_strict'],.5)
        self.assertEqual(a['judge_distinguishes_verifier_ties_fraction'],1.)
        self.assertEqual(verifier_preference(1,1),'TIE')
        r=diagnostic_rates([{'variant_type':'clean_correct','verifier_preference':'TIE'}],'verifier')
        self.assertIsNone(r['better_response_rate']);self.assertEqual(r['wrong_preference_rate'],0.)

    def test_partial_tail_only_is_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_text('{"a":1}\n{"a":')
            self.assertEqual(load_records(p),[{'a':1}]);append_record(p,{'a':2})
            self.assertEqual(load_records(p),[{'a':1},{'a':2}])
            p.write_text('{"a":1}');load_records(p);append_record(p,{'a':2})
            self.assertEqual(len(load_records(p)),2)
            p.write_text('{bad}\n')
            with self.assertRaises(json.JSONDecodeError):load_records(p)

    def test_resume_refuses_changed_inputs(self):
        cfg=load_yaml('configs/feedback.yaml')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';m=p.with_suffix('.meta.json');rows=[{'id':1},{'id':2}]
            resume_records(cfg,rows,p,m,False,'test')
            append_record(p,{'row_index':0,'input_sha256':object_hash(rows[0])})
            self.assertEqual(len(resume_records(cfg,rows,p,m,True,'test')),1)
            with self.assertRaises(ValueError):resume_records(cfg,[{'id':9}],p,m,True,'test')
            with self.assertRaises(ValueError):resume_records(cfg,rows,p,m,False,'test')

    def test_no_silent_config_substitution(self):
        cfg=load_yaml('configs/feedback.yaml');validate_protocol(cfg)
        cfg['math_max_new_tokens']=256
        with self.assertRaises(ValueError):validate_protocol(cfg)

    def test_released_fixed_diagnostic_sensitivity(self):
        cfg=load_yaml('configs/feedback.yaml')
        from common.data import repo_path
        if not repo_path(cfg['paths']['task5_diagnostics']).exists():self.skipTest('Download course assets for fixed-data check')
        load_dataset(cfg,'diagnostics')
        pairs=controlled_pairs(load_diagnostic_groups(cfg['paths']['task5_diagnostics']))
        self.assertEqual(len(pairs),100)
        for r in pairs:
            r['verifier_preference']=verifier_preference(r['a_exact_reward'],r['b_exact_reward'])
            r['judge_preference']='TIE'
        s=summarize_diagnostics(pairs)['sensitivity']['verifier']
        self.assertEqual(s['S_reason'],0.)
        self.assertEqual(s['S_outcome'],1.)
        self.assertEqual(s['distractor_robustness'],1.)

    def test_released_pair_orientation_and_cache(self):
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,messages,**kwargs):return torch.tensor([[1,2]])
            def decode(self,tokens,**kwargs):return 'A'
        class Model:
            calls=0
            def parameters(self):return iter([torch.nn.Parameter(torch.zeros(1))])
            def generate(self,ids,**kwargs):
                self.calls+=1
                assert kwargs['max_new_tokens']==4 and kwargs['do_sample'] is False
                return torch.tensor([[1,2,3]])
        judge=PairwiseAIJudge.__new__(PairwiseAIJudge);judge.cfg={'ai_judge_model':'fixed'}
        judge.tokenizer=Tokenizer();judge.model=Model();judge.cache={}
        with tempfile.TemporaryDirectory() as d:
            judge.cache_path=Path(d)/'cache.json'
            for b in ['one','two','three','four']:
                key=judge._key('Problem','clean',b);swapped=int(key[:8],16)%2==1
                expected='B' if swapped else 'A'
                self.assertEqual(judge.compare('Problem','clean',b),expected)
                calls=judge.model.calls
                self.assertEqual(judge.compare('Problem','clean',b),expected)
                self.assertEqual(judge.model.calls,calls)

    def test_interrupted_cache_rebuilt_from_durable_labels(self):
        cfg=load_yaml('configs/feedback.yaml')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cache=p/'judge_cache.json';cache.write_text('{broken')
            row={'question':'Q','a_response':'A','b_response':'B','judge_preference':'TIE'}
            append_record(p/'diagnostic_pairs.jsonl',row)
            restore_cache(cfg,p,cache)
            stub=PairwiseAIJudge.__new__(PairwiseAIJudge);stub.cfg=cfg
            self.assertEqual(json.loads(cache.read_text())[stub._key('Q','A','B')],'TIE')

    def test_judge_resume_does_not_repeat_completed_rows(self):
        cfg=load_yaml('configs/feedback.yaml')
        class Judge:
            calls=0
            _key=PairwiseAIJudge._key
            def __init__(self,cfg,path):self.cfg=cfg;self.cache=json.loads(path.read_text())
            def compare(self,q,a,b):
                self.__class__.calls+=1
                if self.__class__.calls==2:raise RuntimeError('simulated disconnect')
                return 'A'
        inputs=[{'question':str(i),'a_response':'a','b_response':'b','a_exact_reward':1,'b_exact_reward':0} for i in range(3)]
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PA2_ARTIFACT_ROOT':d}),patch('task5_feedback.judge_pairs.PairwiseAIJudge',Judge):
            p=output_dir(cfg)/'test_pairs.jsonl'
            with self.assertRaises(RuntimeError):score_pairs(cfg,inputs,p,False,'test',smoke=False)
            self.assertEqual(len(load_records(p)),1)
            Judge.calls=2
            result=score_pairs(cfg,inputs,p,True,'test',smoke=False)
            self.assertEqual(len(result),3);self.assertEqual(Judge.calls,4)
            result=score_pairs(cfg,inputs,p,True,'test',smoke=False)
            self.assertEqual(Judge.calls,4)


class GenerationRecoveryTests(unittest.TestCase):
    def test_partial_batch_replays_same_seed_and_membership(self):
        from task5_feedback.generate_responses import generate
        class Tokenizer:
            def apply_chat_template(self,p,**kw):return [1,2,3]
        class Model:
            config=type('Config',(),{'max_position_embeddings':4096})()
            def parameters(self):return iter([torch.nn.Parameter(torch.zeros(1))])
        batches=[]
        def batch_generate(model,tok,prompts,**kw):
            batches.append(len(prompts))
            values=torch.randint(1,99,(len(prompts),))
            return {'responses':[f'#### {v}' for v in values.tolist()],
                    'truncated':[False]*len(prompts),'response_ids':torch.stack([values,torch.zeros_like(values)],dim=1),
                    'response_lengths':[2]*len(prompts),'terminated_with_eos':[True]*len(prompts)}
        rows=[{'question':str(i),'source_index':i,'gold_final':'7','messages':[{'role':'user','content':str(i)}]} for i in range(5)]
        cfg=load_yaml('configs/feedback.yaml')
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PA2_ARTIFACT_ROOT':d}),\
             patch('task5_feedback.generate_responses.load_dataset',return_value=rows),\
             patch('task5_feedback.generate_responses.load_policy',return_value=Model()),\
             patch('task5_feedback.generate_responses.load_tokenizer',return_value=Tokenizer()),\
             patch('task5_feedback.generate_responses.batch_generate',side_effect=batch_generate):
            calls=[0]
            def interrupted_append(p,row):
                calls[0]+=1
                if calls[0]==3:raise RuntimeError('simulated disconnect inside batch')
                append_record(p,row)
            with patch('task5_feedback.generate_responses.append_record',side_effect=interrupted_append):
                with self.assertRaises(RuntimeError):generate(resume=False)
            p=output_dir(cfg)/'gsm_sft_responses.jsonl'
            before=load_records(p);self.assertEqual(len(before),2)
            generate(resume=True);after=load_records(p)
            self.assertEqual(len(after),5);self.assertEqual(before,after[:2]);self.assertEqual(batches,[4,4,1])
            generate(resume=True);self.assertEqual(batches,[4,4,1])



class EvidenceExportTests(unittest.TestCase):
    def test_complete_fixed_set_export_and_smoke_rejection(self):
        from common.data import repo_path
        from task5_feedback.runtime import DOMAINS, POLICIES, finish, code_hash, adapter_identity
        from task5_feedback.compare_feedback import export
        cfg=load_yaml('configs/feedback.yaml')
        if any(not repo_path(cfg['paths'][k]).exists() for k in ('gsm_eval','math_transfer_eval','task5_diagnostics')):
            self.skipTest('Download course data for full evidence aggregation check')
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PA2_ARTIFACT_ROOT':d}):
            out=output_dir(cfg)
            def save(name,inputs,records,kind,**extra):
                p=out/name;m=p.with_suffix('.meta.json')
                resume_records(cfg,inputs,p,m,False,kind,smoke=False,**extra)
                for i,(r,input_row) in enumerate(zip(records,inputs)):
                    r.update(row_index=i,input_sha256=object_hash(input_row));append_record(p,r)
                finish(m,records)
            for domain in DOMAINS:
                inputs=load_dataset(cfg,domain);by_policy={}
                for policy in POLICIES:
                    records=[]
                    for i,r in enumerate(inputs):
                        text=f"#### {r['gold_final']}"
                        from task5_feedback.runtime import row_id
                        records.append({'prompt_id':row_id(r,domain),'question':r['question'],'response':text,
                                        'response_tokens':3,'generation_truncated':False,'prompt_truncated':False,
                                        'generation_seconds_share':.1,**response_metrics(text,r['gold_final'])})
                    by_policy[policy]=records
                    # Adapter identities are checked separately by preflight. This
                    # fixture tests result aggregation, not policy inference.
                    _,sha=adapter_identity(cfg,policy)
                    save(f'{domain}_{policy}_responses.jsonl',inputs,records,'generation',adapter_sha256=sha)
                for policy in ('rlvr','rlaif'):
                    pairs=[{'prompt_id':a['prompt_id'],'dataset':domain,'policy':policy,'question':a['question'],
                            'a_response':a['response'],'b_response':b['response'],
                            'a_exact_reward':a['exact_reward'],'b_exact_reward':b['exact_reward']} for a,b in zip(by_policy[policy],by_policy['sft'])]
                    labels=[{**r,'judge_preference':'TIE','verifier_preference':'TIE',
                             'judge_cache_hit':False,'judge_seconds':.01} for r in pairs]
                    save(f'{domain}_{policy}_pairs.jsonl',pairs,labels,'math_pairwise')
            pairs=controlled_pairs(load_diagnostic_groups(cfg['paths']['task5_diagnostics']))
            labels=[{**r,'judge_preference':'TIE','verifier_preference':verifier_preference(r['a_exact_reward'],r['b_exact_reward']),
                     'judge_cache_hit':False,'judge_seconds':.01} for r in pairs]
            save('diagnostic_pairs.jsonl',pairs,labels,'diagnostic_pairwise')
            export()
            checks=json.loads((out/'completion_checks.json').read_text());self.assertEqual(checks['responses'],1200)
            result=json.loads((out/'comparison.json').read_text())
            self.assertEqual(result['in_domain']['policies']['sft']['exact_accuracy'],1.)
            self.assertEqual(result['transfer']['policies']['rlaif']['ai_tie_adjusted_win_rate_vs_sft'],.5)
            self.assertEqual(result['diagnostics']['sensitivity']['verifier']['S_outcome'],1.)
            self.assertEqual(len(json.loads((out/'qualitative_diagnostics.json').read_text())),4)
            self.assertTrue((out/'summary.csv').exists())
            from task5_feedback.plot_results import main as plot
            with patch('sys.argv',['plot_results']):plot()
            self.assertEqual(len(list((out/'plots').glob('*.png'))),3)
            self.assertEqual(len(list((out/'plots').glob('*.pdf'))),3)
            p=out/'gsm_sft_responses.meta.json';meta=json.loads(p.read_text());meta['smoke']=True;write_json(p,meta)
            with self.assertRaises(ValueError):export()



class TensorGenerationTests(unittest.TestCase):
    def test_real_tiny_qwen_generation_response_masks(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from common.generation import batch_generate
        class Tokenizer:
            pad_token_id=0
            eos_token_id=2
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
            def __call__(self,texts,**kwargs):
                ids=[[int(n) for n in t.split()] for t in texts];width=max(map(len,ids))
                return {'input_ids':torch.tensor([[0]*(width-len(r))+r for r in ids]),
                        'attention_mask':torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in ids])}
            def decode(self,tokens,**kwargs):return ' '.join(map(str,tokens.tolist()))
        torch.manual_seed(6304)
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
                       eos_token_id=2,pad_token_id=0)
        model=Qwen2ForCausalLM(cfg).eval()
        result=batch_generate(model,Tokenizer(),[[{'role':'user','content':'3 4'}],
                                                [{'role':'user','content':'3 4 5'}]],
                              max_prompt_length=3,max_new_tokens=4,temperature=.7,top_p=.9,do_sample=True)
        self.assertEqual(result['sequences'][:,:3].tolist(),[[0,3,4],[3,4,5]])
        self.assertEqual(len(result['responses']),2)
        self.assertTrue(all(1<=n<=4 for n in result['response_lengths']))
        for n,mask in zip(result['response_lengths'],result['response_mask']):
            self.assertEqual(int(mask.sum()),n)

if __name__=='__main__':unittest.main()

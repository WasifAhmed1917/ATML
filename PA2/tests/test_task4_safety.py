"""Safety denominators, blind human work, checkpoint selection, and disconnect recovery."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
import torch
from common.data import load_yaml
from task4_safety import runtime as rt
from task4_safety.evaluate_safety import summarize,category_distributions,audit_agreement,evaluate
from task4_safety.generate_responses import generate_for_policy
from task4_safety.judge_responses import parse_json,judge_one,JUDGE_PROMPT,run_judging
from task4_safety.make_audit_sheet import fixed_audit_ids,make_audit


def row(i=0,p='sft',b='SAFE',label='SAFE_ANSWER'):
    return dict(row_index=i,xstest_id=i,policy=p,prompt=f'prompt {i}',benchmark_class=b,type='category',response=f'response {i}',
                response_tokens=3,truncated=False,prompt_truncated=False,label=label,confidence=.8,rationale_tag='test')


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.env=patch.dict(os.environ,{'PA2_ARTIFACT_ROOT':self.tmp.name});self.env.start();self.cfg=load_yaml('configs/feedback.yaml')
    def tearDown(self):self.env.stop();self.tmp.cleanup()

    def test_conditional_denominators_and_all_categories(self):
        rows=[row(i,b=b,label=l) for i,(b,l) in enumerate([('SAFE','SAFE_ANSWER'),('SAFE','OVER_REFUSAL'),('SAFE','AMBIGUOUS'),('UNSAFE','UNSAFE_COMPLIANCE'),('UNSAFE','JUSTIFIED_REFUSAL')])]
        s=summarize(rows);self.assertAlmostEqual(s['safe_answer_rate'],1/3);self.assertAlmostEqual(s['safe_over_refusal_rate'],1/3)
        self.assertEqual(s['unsafe_compliance_rate'],.5);self.assertEqual(s['unsafe_justified_refusal_rate'],.5);self.assertEqual(s['ambiguous_judge_rate'],.2)
        rows[0]['confidence']=0.;self.assertEqual(s,summarize(rows));d=category_distributions(rows);self.assertEqual(len(d),10)
        self.assertAlmostEqual(sum(r['rate'] for r in d if r['benchmark_class']=='SAFE'),1.)

    def test_exact_judge_prompt_parser_and_generation_arguments(self):
        self.assertEqual(parse_json('not json')['rationale_tag'],'parse_failure')
        self.assertEqual(parse_json('{"label":"unknown","confidence":2}')['label'],'AMBIGUOUS')
        self.assertEqual(parse_json('{"label":"safe_answer","confidence":2}')['confidence'],1.)
        tok=MagicMock();tok.apply_chat_template.return_value=torch.tensor([[1,2]]);tok.eos_token_id=0;tok.decode.return_value='{"label":"SAFE_ANSWER","confidence":0.7}'
        model=MagicMock();model.parameters.return_value=iter([torch.nn.Parameter(torch.zeros(1))]);model.generate.return_value=torch.tensor([[1,2,3]])
        self.assertEqual(judge_one(tok,model,'benign','answer')['label'],'SAFE_ANSWER')
        kw=model.generate.call_args.kwargs;self.assertEqual(kw['max_new_tokens'],64);self.assertFalse(kw['do_sample'])
        self.assertEqual(tok.apply_chat_template.call_args.args[0][0]['content'],JUDGE_PROMPT.format(prompt='benign',response='answer'))

    def test_protocol_and_checkpoint_reject_smoke_or_ablation(self):
        rt.validate_protocol(self.cfg)
        for k,v in [('safety_max_new_tokens',768),('manual_audit_per_class',10),('quantize_frozen_models',False)]:
            cfg=copy.deepcopy(self.cfg);cfg[k]=v
            with self.assertRaises(ValueError):rt.validate_protocol(cfg)
        cfg=copy.deepcopy(self.cfg);cfg['policies']['ppo']='outputs/task2_ppo/clip_0.20'
        with self.assertRaises(ValueError):rt.validate_protocol(cfg)
        out=rt.artifact_path(self.cfg['policies']['ppo']);out.mkdir(parents=True)
        meta=dict(status='complete',smoke=False,run_name='standard',updates=20,completed_updates=20,config={'base_model':self.cfg['base_model']})
        rt.write_json(out/'training_metadata.json',meta);(out/'adapter_model.safetensors').write_bytes(b'test');rt.write_json(out/'adapter_config.json',{'base_model_name_or_path':self.cfg['base_model']})
        rt.validate_adapter(self.cfg,'ppo');meta['smoke']=True;rt.write_json(out/'training_metadata.json',meta)
        with self.assertRaises(ValueError):rt.validate_adapter(self.cfg,'ppo')

    def test_fixed_selection_matches_released_rng(self):
        rows=[row(i,b='SAFE' if i<40 else 'UNSAFE') for i in range(80)];rng=np.random.default_rng(6304)
        expected=sorted(rng.choice(np.arange(40),30,replace=False).tolist()+rng.choice(np.arange(40,80),30,replace=False).tolist())
        self.assertEqual(fixed_audit_ids(rows,30,6304),expected)

    def test_manual_agreement_includes_ambiguity_and_partial_status(self):
        rows=[row(i,p,b='SAFE' if i==0 else 'UNSAFE',label='SAFE_ANSWER' if i==0 else 'JUSTIFIED_REFUSAL') for p in rt.POLICIES for i in range(2)]
        f=pd.DataFrame(rows).drop(columns=['label','confidence','rationale_tag']);f['manual_label']='';sheet=Path(self.tmp.name)/'audit.csv';f.to_csv(sheet,index=False)
        result,cm,_=audit_agreement(rows,sheet,[0,1]);self.assertEqual(result['status'],'pending_manual_labels');self.assertIsNone(result['summaries'][0]['agreement'])
        f['manual_label']=[r['label'] for r in rows];f.loc[0,'manual_label']='AMBIGUOUS';f.to_csv(sheet,index=False)
        result,cm,_=audit_agreement(rows,sheet,[0,1]);self.assertEqual(result['status'],'complete');self.assertEqual(result['summaries'][0]['agreement'],7/8);self.assertEqual(result['summaries'][0]['manual_ambiguous_rate'],1/8)
        self.assertEqual(sum(r['count'] for r in cm if r['policy']=='all'),8)
        f.loc[0,'manual_label']='bad';f.to_csv(sheet,index=False)
        with self.assertRaises(ValueError):audit_agreement(rows,sheet,[0,1])

    def test_generation_replays_original_batch_after_partial_append(self):
        data=pd.DataFrame([row(i) for i in range(6)]);tok=MagicMock();tok.apply_chat_template.return_value=[1,2];model=MagicMock();batches=[]
        def gen(model,tok,prompts,**kw):
            batches.append([p[0]['content'] for p in prompts]);self.assertFalse(kw['do_sample']);self.assertEqual(kw['max_new_tokens'],256)
            n=len(prompts);return dict(responses=[f'answer {p[0]["content"]}' for p in prompts],response_lengths=[2]*n,response_ids=torch.tensor([[3,0]]*n),terminated_with_eos=[True]*n,truncated=[False]*n)
        original=rt.append_record;calls=[0]
        def disconnect(path,r):
            if calls[0]==2:raise RuntimeError('disconnect')
            original(path,r);calls[0]+=1
        with patch('task4_safety.generate_responses.load_xstest',return_value=data),patch('task4_safety.generate_responses.load_tokenizer',return_value=tok),patch('task4_safety.generate_responses.load_policy',return_value=model),patch('task4_safety.generate_responses.batch_generate',side_effect=gen):
            with patch('task4_safety.generate_responses.append_record',side_effect=disconnect):
                with self.assertRaises(RuntimeError):generate_for_policy(self.cfg,'sft',4,True)
            rows=generate_for_policy(self.cfg,'sft',4,True);self.assertEqual([r['xstest_id'] for r in rows],list(range(6)));self.assertEqual(batches[0],batches[1]);self.assertEqual(batches[2],['prompt 4','prompt 5'])
            with patch('task4_safety.generate_responses.load_policy',side_effect=AssertionError('must skip')):generate_for_policy(self.cfg,'sft',4,True)
            with self.assertRaises(ValueError):generate_for_policy(self.cfg,'sft',2,True)

    def test_judge_resume_skips_saved_responses(self):
        out=rt.output_dir(self.cfg);out.mkdir(parents=True)
        for i in range(3):rt.append_record(out/'generated_sft.jsonl',row(i))
        rt.write_json(out/'generation_sft_metadata.json',{'status':'complete','smoke':False});model=MagicMock();model.parameters.return_value=iter([torch.nn.Parameter(torch.zeros(1))]);calls=[]
        def judge(tok,model,prompt,response,limit):
            calls.append((prompt,response,limit))
            if len(calls)==2:raise RuntimeError('disconnect')
            return dict(label='SAFE_ANSWER',confidence=.8,rationale_tag='test')
        with patch('task4_safety.judge_responses.load_judge',return_value=(None,model)),patch('task4_safety.judge_responses.judge_one',side_effect=judge):
            with self.assertRaises(RuntimeError):run_judging(self.cfg,'sft',True)
        with patch('task4_safety.judge_responses.load_judge',return_value=(None,model)),patch('task4_safety.judge_responses.judge_one',return_value=dict(label='AMBIGUOUS',confidence=0.,rationale_tag='test')) as scorer:
            self.assertEqual(len(run_judging(self.cfg,'sft',True)),3);self.assertEqual(scorer.call_count,2)
        with patch('task4_safety.judge_responses.load_judge',side_effect=AssertionError('must skip')):run_judging(self.cfg,'sft',True)

    def test_full_export_blind_audit_preservation_and_plots(self):
        data=pd.DataFrame([row(i,b='SAFE' if i<40 else 'UNSAFE') for i in range(80)]);cfg=self.cfg;out=rt.output_dir(cfg);out.mkdir(parents=True);inputs=data.to_dict('records')
        for policy in rt.POLICIES:
            generated=[]
            for r in inputs:
                r={**r,'policy':policy};r.pop('label');r.pop('confidence');r.pop('rationale_tag');generated.append(r);rt.append_record(out/f'generated_{policy}.jsonl',r)
                rt.append_record(out/f'judged_{policy}.jsonl',{**r,'label':'SAFE_ANSWER' if r['benchmark_class']=='SAFE' else 'JUSTIFIED_REFUSAL','confidence':.7,'rationale_tag':'test'})
            decoding=dict(do_sample=False,max_prompt_length=256,max_new_tokens=256,temperature=0.,top_p=1.)
            rt.write_json(out/f'generation_{policy}_metadata.json',dict(status='complete',smoke=False,batch_size=4,signature=rt.signature(cfg,inputs,kind='generation',policy=policy,adapter_sha256=None,smoke=False,batch_size=4,decoding=decoding)))
            rt.write_json(out/f'judge_{policy}_metadata.json',dict(status='complete',smoke=False,signature=rt.signature(cfg,generated,kind='judge',policy=policy,smoke=False,generation_sha256=rt.file_hash(out/f'generated_{policy}.jsonl'),judge_prompt=JUDGE_PROMPT)))
        make_audit(cfg);sheet=out/'manual_audit_blind.csv';f=pd.read_csv(sheet,keep_default_na=False);self.assertEqual(len(f),240);self.assertNotIn('label',f);self.assertNotIn('confidence',f)
        f.loc[0,'manual_label']='AMBIGUOUS';f.to_csv(sheet,index=False);make_audit(cfg);self.assertEqual(pd.read_csv(sheet,keep_default_na=False).loc[0,'manual_label'],'AMBIGUOUS')
        with patch('task4_safety.generate_responses.load_xstest',return_value=data),patch('task4_safety.runtime.validate_adapter',return_value=(None,None)):
            summary,checks=evaluate();self.assertFalse(checks['manual_audit_complete']);self.assertEqual(summary[0]['safe_answer_rate'],1.)
            f['manual_label']=np.where(f.benchmark_class=='SAFE','SAFE_ANSWER','JUSTIFIED_REFUSAL');f.to_csv(sheet,index=False);_,checks=evaluate();self.assertTrue(all(checks.values()))
            from task4_safety.plot_results import plot
            plot();self.assertTrue((out/'plots/audit_confusion.png').exists())
        f.loc[0,'response']='modified';f.to_csv(sheet,index=False)
        with self.assertRaises(ValueError):make_audit(cfg)

    def test_real_tiny_qwen_greedy_generation_and_frozen_judge(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from common.generation import batch_generate
        backend=Tokenizer(WordLevel({'<eos>':0,'<pad>':1,'<unk>':2,'user':3,'assistant':4,'hello':5,'world':6},unk_token='<unk>'))
        backend.pre_tokenizer=Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<pad>',unk_token='<unk>',padding_side='left')
        tok.model_input_names=['input_ids','attention_mask']
        tok.chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=7,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,eos_token_id=0,pad_token_id=1))
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        prompts=[[{'role':'user','content':'hello'}],[{'role':'user','content':'hello world'}]]
        a=batch_generate(model,tok,prompts,256,4,temperature=0.,top_p=1.,do_sample=False)
        b=batch_generate(model,tok,prompts,256,4,temperature=0.,top_p=1.,do_sample=False)
        self.assertTrue(torch.equal(a['response_ids'],b['response_ids']));self.assertTrue(all(n<=4 for n in a['response_lengths']))
        label=judge_one(tok,model,'hello','world',4);self.assertIn(label['label'],rt.LABEL_ORDER)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

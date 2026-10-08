import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen2ForSequenceClassification
from peft import get_peft_model,LoraConfig
from task2_ppo.runtime import (token_statistics,response_values,joint_update,parameter_state,
                              load_parameter_state,atomic_torch_save)
from task2_ppo.continue_train import update_on_rollout
from common.models import value_parameter_groups

class PPOPipelineTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(6304)
        cfg=Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                        num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,num_labels=1)
        policy=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        value=get_peft_model(Qwen2ForSequenceClassification(cfg),LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='SEQ_CLS',modules_to_save=['score']))
        policy.eval();value.eval();return policy,value

    def test_actual_ppo_policy_critic_and_alignment(self):
        policy,value=self.models();ids=torch.tensor([[3,4,5,6,7]])
        r={'sequences':ids,'attention_mask':torch.ones_like(ids),'prompt_width':2,'response_ids':ids[:,2:],
           'response_mask':torch.ones(1,3)}
        with torch.no_grad():
            old,e=token_statistics(policy,r,entropy=True);v=response_values(value,r)
            lp=policy(input_ids=ids).logits.float().log_softmax(-1)
            expected=lp[:,1:4].gather(-1,ids[:,2:].unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(old,expected)
            self.assertEqual(tuple(v.shape),(1,3));self.assertTrue((e>0).all())
        r.update({'old_logprobs':old,'ref_logprobs':old,'values':v,'effective_terminal_reward':torch.tensor([2.])})
        before=parameter_state(policy);vb=parameter_state(value)
        bundle={'policy':policy,'value_model':value,'policy_optimizer':torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-3),
                'value_optimizer':torch.optim.AdamW(value_parameter_groups(value,1e-3,1e-3)),
                'cfg':{'kl_beta':.1,'gamma':1.,'gae_lambda':.95,'ppo_epochs':2,'clip_epsilon':.2,'value_coef':.5,'max_grad_norm':1.}}
        d=update_on_rollout(bundle,r,torch.amp.GradScaler('cpu',enabled=False))
        self.assertEqual(len(d),2);self.assertEqual(d[0]['clip_fraction'],0.)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in parameter_state(policy).items()))
        self.assertTrue(any(not torch.equal(vb[n],p) for n,p in parameter_state(value).items()))
        # Save/restore both trainable models and both optimizer states, then match an uninterrupted next update.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            atomic_torch_save(path,{'policy':parameter_state(policy),'value':parameter_state(value),
                                   'po':bundle['policy_optimizer'].state_dict(),'vo':bundle['value_optimizer'].state_dict()})
            p2,v2=self.models();cp=torch.load(path,weights_only=False)
            load_parameter_state(p2,cp['policy']);load_parameter_state(v2,cp['value'])
            po2=torch.optim.AdamW([p for p in p2.parameters() if p.requires_grad],lr=1e-3)
            vo2=torch.optim.AdamW(value_parameter_groups(v2,1e-3,1e-3));po2.load_state_dict(cp['po']);vo2.load_state_dict(cp['vo'])
            b2={**bundle,'policy':p2,'value_model':v2,'policy_optimizer':po2,'value_optimizer':vo2}
            update_on_rollout(bundle,r,torch.amp.GradScaler('cpu',enabled=False));update_on_rollout(b2,r,torch.amp.GradScaler('cpu',enabled=False))
            for a,b in zip(parameter_state(policy).values(),parameter_state(p2).values()):torch.testing.assert_close(a,b)
            for a,b in zip(parameter_state(value).values(),parameter_state(v2).values()):torch.testing.assert_close(a,b)

    def test_training_checkpoint_completes_after_interrupted_final_save(self):
        from task2_ppo.continue_train import run_ppo
        class Tokenizer:
            def save_pretrained(self,path):pass
        cfg={'seed':6304,'base_model':'tiny','dtype':'float32','updates':20,'prompts_per_update':1,
             'max_response_length':512,'ppo_epochs':2,'gamma':1.,'gae_lambda':.95,'value_coef':.5,
             'max_grad_norm':1.,'clip_epsilon':.2,'kl_beta':.1,'policy_learning_rate':1e-3,
             'paths':{'rl_prompt_train':'unused','ppo_midpoint_policy':'unused','ppo_midpoint_value':'unused'}}
        row={'prompt_id':'fixed-id','source_index':1,'messages':[{'role':'user','content':'prompt'}]}
        def prepare(config):
            policy,value=self.models()
            return {'cfg':config,'policy':policy,'value_model':value,'tokenizer':Tokenizer(),
                    'policy_optimizer':torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-3),
                    'value_optimizer':torch.optim.AdamW(value_parameter_groups(value,1e-3,1e-3))}
        def collect(bundle,rows):
            ids=torch.tensor([[3,4,5,6,7]])
            r={'sequences':ids,'attention_mask':torch.ones_like(ids),'prompt_width':2,'response_ids':ids[:,2:],
               'response_mask':torch.ones(1,3),'rows':rows,'messages':[row['messages']],
               'responses':['text'],'terminated_with_eos':[True],'truncated':[False]}
            with torch.no_grad():old,_=token_statistics(bundle['policy'],r);v=response_values(bundle['value_model'],r)
            r.update({'old_logprobs':old,'ref_logprobs':old,'values':v,'raw_terminal_reward':torch.tensor([2.]),'effective_terminal_reward':torch.tensor([2.])})
            return r
        with tempfile.TemporaryDirectory() as tmp, patch('task2_ppo.continue_train.load_yaml',side_effect=lambda _:dict(cfg)), \
             patch('task2_ppo.continue_train.read_jsonl',return_value=[row]), \
             patch('task2_ppo.continue_train.prepare_ppo_continuation',side_effect=prepare), \
             patch('task2_ppo.continue_train.collect_rollout',side_effect=collect):
            out=Path(tmp)/'run'
            # Crash after the update checkpoint has been committed, before final adapter save.
            with patch('peft.PeftModel.save_pretrained',side_effect=RuntimeError('simulated disconnect')):
                with self.assertRaises(RuntimeError):run_ppo(output=str(out),run_name='smoke',smoke=True)
            checkpoint=torch.load(out/'continuation.pt',weights_only=False)
            self.assertEqual(checkpoint['completed_updates'],1)
            with patch('task2_ppo.continue_train.collect_rollout',side_effect=AssertionError('completed rollout replayed')):
                meta=run_ppo(output=str(out),run_name='smoke',smoke=True,resume=True)
            self.assertEqual(meta['status'],'complete');self.assertEqual(meta['completed_updates'],1)
            self.assertTrue((out/'adapter_model.safetensors').exists())
            self.assertTrue((out/'critic_adapter/adapter_model.safetensors').exists())
            self.assertEqual(len(json.loads((out/'training_history.json').read_text())),1)
            with patch('task2_ppo.continue_train.prepare_ppo_continuation',side_effect=AssertionError('completed model loaded')):
                run_ppo(output=str(out),run_name='smoke',smoke=True,resume=True,skip_completed=True)

    def test_fp16_base_with_fp32_critic_head(self):
        policy,value=self.models()
        for model in (policy,value):
            model.half()
            for p in model.parameters():
                if p.requires_grad:p.data=p.data.float()
        ids=torch.tensor([[3,4,5,6,7]])
        r={'sequences':ids,'attention_mask':torch.ones_like(ids),'prompt_width':2,'response_ids':ids[:,2:],
           'response_mask':torch.ones(1,3)}
        result=response_values(value,r)
        self.assertEqual(result.dtype,torch.float32)
        result.sum().backward()
        self.assertTrue(all(p.grad is None or p.grad.dtype==torch.float32 for p in value.parameters() if p.requires_grad))

    def test_joint_overflow_no_partial_optimizer_step(self):
        a=torch.nn.Linear(1,1,bias=False);b=torch.nn.Linear(1,1,bias=False)
        opts=[torch.optim.SGD(a.parameters(),lr=.1),torch.optim.SGD(b.parameters(),lr=.1)]
        initial=a.weight.detach().clone();scale=torch.amp.GradScaler('cpu',init_scale=8.)
        draws=[]
        def backward():
            draws.append(torch.rand(1));loss=a.weight.sum()+b.weight.sum()
            scale.scale(loss).backward()
            if len(draws)==1:b.weight.grad.fill_(float('inf'))
            return {}
        _,_,retries,_=joint_update([a,b],opts,scale,backward,max_grad_norm=100.)
        self.assertEqual(retries,1);torch.testing.assert_close(draws[0],draws[1])
        torch.testing.assert_close(a.weight,initial-.1)
        initial=a.weight.detach().clone()
        def broken():
            scale.scale(a.weight.sum()+b.weight.sum()).backward();b.weight.grad.fill_(float('inf'));return {}
        with self.assertRaises(FloatingPointError):joint_update([a,b],opts,scale,broken,max_retries=1)
        torch.testing.assert_close(a.weight,initial)
if __name__=='__main__':unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from peft import get_peft_model,LoraConfig
from task3_grpo.grpo import grpo_policy_loss,group_relative_advantages
from task3_grpo.runtime import token_statistics,update_on_rollout,parameter_state,load_parameter_state,atomic_torch_save,group_diagnostics


class GRPOPipelineTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(6304)
        cfg=Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=2,pad_token_id=0)
        model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,
                             target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.eval();return model

    def rollout(self,model):
        ids=torch.tensor([[3,4,5,6,0],[3,4,7,8,9],[3,4,10,11,12],[3,4,13,14,15]])
        mask=torch.tensor([[1.,1.,0.],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]])
        r={'sequences':ids,'attention_mask':torch.cat([torch.ones(4,2),mask],dim=1).long(),
           'prompt_width':2,'response_ids':ids[:,2:],'response_mask':mask}
        with torch.no_grad():old,en=token_statistics(model,r,entropy=True)
        training=mask.clone();training[2]=0
        rewards=torch.tensor([1.,3.,2.,4.]);groups=torch.tensor([0,0,1,1])
        r.update({'old_logprobs':old,'ref_logprobs':old-.02,'training_mask':training,
                  'advantages':group_relative_advantages(rewards,groups),'rewards':rewards,'group_ids':groups,'old_entropy':en})
        return r

    def bundle(self,model):
        return {'policy':model,'optimizer':torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3),
            'cfg':{'policy_epochs':1,'clip_epsilon':.2,'kl_beta':.1,'max_completion_length':3,'max_grad_norm':1e6}}

    def test_actual_microbatch_update_matches_full_batch_both_normalizations(self):
        for kind in ('grpo','dr_grpo'):
            model=self.model();other=self.model();r=self.rollout(model);bundle=self.bundle(model);full=self.bundle(other)
            new,_=token_statistics(other,r,entropy=True)
            loss,_=grpo_policy_loss(new,r['old_logprobs'],r['advantages'],r['training_mask'],r['ref_logprobs'],.2,.1,kind,3)
            loss.backward();full['optimizer'].step()
            d=update_on_rollout(bundle,r,torch.amp.GradScaler('cpu',enabled=False),kind)
            self.assertAlmostEqual(d[0]['loss'],float(loss.detach()),places=6)
            for a,b in zip(parameter_state(model).values(),parameter_state(other).values()):torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)
            self.assertEqual(len(d[0]['length_statistics']),3)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(parameter_state(self.model()).values(),parameter_state(model).values())))

    def test_group_collection_clones_inference_tensors_and_preserves_prompt_groups(self):
        from task3_grpo.runtime import collect_group_rollout
        model=self.model();template=self.rollout(model)
        with torch.inference_mode():
            generated={k:v.repeat((2,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v
                       for k,v in template.items() if k in ('sequences','attention_mask','response_ids','response_mask','prompt_width')}
        generated.update({'responses':['r']*8,'terminated_with_eos':[True,True,False,True]*2,
                          'truncated':[False,False,True,False]*2})
        rows=[{'prompt_id':str(i),'source_index':i,'messages':[{'role':'user','content':'p'}]} for i in range(2)]
        bundle={'policy':model,'tokenizer':None,'reward_model':None,'reward_tokenizer':None,
                'cfg':{'num_generations':4,'max_prompt_length':2,'max_completion_length':3,
                       'generation':{},'reward_max_length':5,'mask_truncated_completions':True}}
        rewards=torch.tensor([1.,3.,2.,4.,100.,102.,101.,103.])
        with patch('task3_grpo.runtime.batch_generate',return_value=generated), \
             patch('task3_grpo.runtime.score_reward_pairs',return_value=rewards):
            r=collect_group_rollout(bundle,rows)
        torch.testing.assert_close(r['group_ids'],torch.tensor([0]*4+[1]*4))
        torch.testing.assert_close(r['advantages'][:4],r['advantages'][4:])
        self.assertEqual(int(r['training_mask'][2].sum()),0)
        self.assertFalse(r['sequences'].is_inference());self.assertFalse(r['old_logprobs'].requires_grad)
        bundle.update({'optimizer':self.bundle(model)['optimizer']})
        bundle['cfg'].update({'policy_epochs':1,'clip_epsilon':.2,'kl_beta':.1,'max_grad_norm':1.})
        d=update_on_rollout(bundle,r,torch.amp.GradScaler('cpu',enabled=False))
        self.assertTrue(d[0]['gradient_norm']>0)

    def test_all_truncated_has_no_weight_decay_and_zero_effective_signal(self):
        model=self.model();r=self.rollout(model);r['training_mask'].zero_();before=parameter_state(model)
        bundle=self.bundle(model);d=update_on_rollout(bundle,r,torch.amp.GradScaler('cpu',enabled=False))
        self.assertTrue(d[0]['skipped_all_truncated']);self.assertEqual(bundle['optimizer'].state,{})
        for n,p in parameter_state(model).items():torch.testing.assert_close(before[n],p)
        self.assertEqual(group_diagnostics(r)['effective_informative_group_fraction'],0.)

    def test_fp16_overflow_replays_same_epoch_without_partial_step(self):
        from task3_grpo.runtime import joint_update
        model=torch.nn.Linear(1,1,bias=False);opt=torch.optim.SGD(model.parameters(),lr=.1)
        initial=model.weight.detach().clone();scaler=torch.amp.GradScaler('cpu',init_scale=8.);draws=[]
        def backward():
            draws.append(torch.rand(1));scaler.scale(model.weight.sum()).backward()
            if len(draws)==1:model.weight.grad.fill_(float('inf'))
            return {}
        _,_,retries,_=joint_update([model],[opt],scaler,backward,max_grad_norm=100.)
        self.assertEqual(retries,1);torch.testing.assert_close(draws[0],draws[1])
        torch.testing.assert_close(model.weight,initial-.1)

    def test_optimizer_restore_matches_uninterrupted_next_update(self):
        model=self.model();r=self.rollout(model);bundle=self.bundle(model);scaler=torch.amp.GradScaler('cpu',enabled=False)
        update_on_rollout(bundle,r,scaler)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cp.pt';atomic_torch_save(path,{'policy':parameter_state(model),'optimizer':bundle['optimizer'].state_dict()})
            cp=torch.load(path,weights_only=False);other=self.model();load_parameter_state(other,cp['policy']);second=self.bundle(other)
            second['optimizer'].load_state_dict(cp['optimizer'])
            update_on_rollout(bundle,r,scaler);update_on_rollout(second,r,scaler)
            for a,b in zip(parameter_state(model).values(),parameter_state(other).values()):torch.testing.assert_close(a,b)

    def test_resume_after_interrupt_during_final_adapter_save(self):
        from task3_grpo.continue_train import run_grpo
        class Tokenizer:
            def save_pretrained(self,path):pass
        row={'prompt_id':'fixed','source_index':1,'messages':[{'role':'user','content':'p'}]}
        cfg={'seed':6304,'dtype':'float32','updates':20,'prompts_per_update':1,'num_generations':4,
             'max_completion_length':512,'output':'unused','paths':{'rl_prompt_train':'unused','grpo_midpoint_policy':'unused'}}
        def prepare(config):
            b=self.bundle(self.model());b['cfg']={**b['cfg'],**config};b['tokenizer']=Tokenizer();return b
        def collect(bundle,rows):
            r=self.rollout(bundle['policy']);r.update({'rows':[row]*4,'messages':[row['messages']]*4,
                'responses':['r']*4,'terminated_with_eos':[True,True,False,True],'truncated':[False,False,True,False]});return r
        with tempfile.TemporaryDirectory() as tmp, patch('task3_grpo.continue_train.load_yaml',side_effect=lambda _:dict(cfg)), \
             patch('task3_grpo.continue_train.read_jsonl',return_value=[row]), \
             patch('task3_grpo.continue_train.prepare_grpo_continuation',side_effect=prepare), \
             patch('task3_grpo.continue_train.collect_group_rollout',side_effect=collect):
            out=Path(tmp)/'run'
            with patch('peft.PeftModel.save_pretrained',side_effect=RuntimeError('disconnect')):
                with self.assertRaises(RuntimeError):run_grpo(output=str(out),run_name='smoke',smoke=True)
            self.assertEqual(torch.load(out/'continuation.pt',weights_only=False)['completed_updates'],1)
            with patch('task3_grpo.continue_train.collect_group_rollout',side_effect=AssertionError('replayed committed update')):
                meta=run_grpo(output=str(out),run_name='smoke',smoke=True,resume=True)
            self.assertEqual(meta['completed_updates'],1);self.assertEqual(meta['status'],'complete')
            self.assertTrue((out/'adapter_model.safetensors').exists())
            with patch('task3_grpo.continue_train.prepare_grpo_continuation',side_effect=AssertionError('loaded completed model')):
                run_grpo(output=str(out),run_name='smoke',smoke=True,resume=True,skip_completed=True)
if __name__=='__main__':unittest.main()

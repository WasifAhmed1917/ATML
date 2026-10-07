"""CPU integration with a real tiny Qwen2 + PEFT model and fixture data.

Fixture sizes/generation cap are not course experiments or report results.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import yaml
from peft import get_peft_model,PeftModel
from transformers import Qwen2Config,Qwen2ForCausalLM
from common.models import make_lora_config,reference_mode
from task1_dpo import train,evaluate
from task1_dpo.utils import load_records,sequence_logprobs


class TinyTokenizer:
    pad_token_id=0;eos_token_id=2;padding_side='left'
    def encode(self,text):return [3+int(hashlib.sha256(w.encode()).hexdigest()[:6],16)%29 for w in text.split()]
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=' '.join(m['role']+' '+m['content'] for m in messages)
        if add_generation_prompt:text+=' assistant'
        return self.encode(text) if tokenize else text
    def __call__(self,text,add_special_tokens=False,return_tensors=None,padding=False,truncation=False,max_length=None):
        single=isinstance(text,str);texts=[text] if single else text
        rows=[self.encode(s) for s in texts]
        if truncation:rows=[r[:max_length] for r in rows]
        if return_tensors:
            width=max(map(len,rows));ids=[[0]*(width-len(r))+r for r in rows]
            masks=[[0]*(width-len(r))+[1]*len(r) for r in rows]
            return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks)}
        return {'input_ids':rows[0] if single else rows}
    def decode(self,ids,skip_special_tokens=True):
        return ' '.join('token'+str(int(i)) for i in ids if int(i)>2)
    def save_pretrained(self,path):
        Path(path,'fixture_tokenizer.json').write_text('{}')


def tiny_policy(cfg,adapter_path=None,trainable=False,fresh_lora=False):
    with torch.random.fork_rng():
        torch.manual_seed(777)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=40,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
            pad_token_id=0,eos_token_id=2,bos_token_id=1))
    if adapter_path:model=PeftModel.from_pretrained(model,adapter_path,is_trainable=trainable)
    elif fresh_lora:model=get_peft_model(model,make_lora_config(cfg))
    if trainable:
        model.train();model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.eval()
    return model


class TinyReward(torch.nn.Module):
    def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
    def forward(self,input_ids,attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(logits=attention_mask.float().sum(-1,keepdim=True)/10)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_train_evaluate_resume_and_frozen_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[{'prompt_id':f'p{i}','source_index':i,'source_split':'fixture',
                'prompt':'Explain learning in at most 8 words.',
                'chosen':'Learning adapts weights using data.',
                'rejected':'An unrelated lengthy sentence about another subject.',
                'length_stratum':['preferred_longer','length_matched','rejected_longer'][i%3]} for i in range(4)]
            data=root/'pairs.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            words=root/'words.jsonl';words.write_text(json.dumps({'prompt_id':'wl1','messages':[
                {'role':'user','content':'Explain learning in at most 8 words.'}]})+'\n')
            cfg={'seed':6304,'base_model':'fixture','dtype':'float32','quantize_frozen_models':False,
                'lora':{'r':2,'alpha':4,'dropout':0.05,'target_modules':['q_proj','v_proj']},
                'generation':{'do_sample':True,'temperature':.7,'top_p':.9},
                'beta':.1,'batch_size':2,'grad_accum_steps':2,'learning_rate':2e-5,'weight_decay':0.,
                'epochs':1,'max_sequence_length':64,'max_generation_tokens':8,'max_grad_norm':1.,
                'standard_output':str(root/'adapter'),'results_dir':str(root/'results'),
                'paths':{'dpo_standard_train':str(data),'dpo_standard_eval':str(data),
                         'dpo_length_eval':str(data),'word_limit_prompts':str(words)}}
            config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg))
            with patch.object(train,'load_policy',tiny_policy),patch.object(train,'load_tokenizer',lambda _:TinyTokenizer()),\
                 patch.object(evaluate,'load_policy',tiny_policy),patch.object(evaluate,'load_tokenizer',lambda _:TinyTokenizer()),\
                 patch.object(evaluate,'load_reward_model',lambda _: (TinyReward(),TinyTokenizer())):
                out=train.run_training(str(config),smoke=True)
                self.assertEqual(json.loads((out/'training_metadata.json').read_text())['optimizer_steps'],1)
                trained=tiny_policy(cfg,str(out));base=tiny_policy(cfg)
                batch=train.make_collate(TinyTokenizer(),64)(rows[:1])[0]
                with torch.no_grad(),reference_mode(trained):a=sequence_logprobs(trained,batch)[0]
                with torch.no_grad():b=sequence_logprobs(base,batch)[0]
                torch.testing.assert_close(a,b)
                s=evaluate.run_evaluation(str(config),str(out),'standard',length=True,smoke=True)
                self.assertEqual(s['n_pairs'],2);self.assertEqual(s['n_prompts'],2)
                before=(root/'results/standard/generations.jsonl').read_bytes()
                s2=evaluate.run_evaluation(str(config),str(out),'standard',length=True,resume=True,smoke=True)
                self.assertEqual(s,s2)
                self.assertEqual(before,(root/'results/standard/generations.jsonl').read_bytes())
                self.assertEqual(train.run_training(str(config),skip_completed=True,smoke=True),out)
                # Simulate interruption after checkpoint, before the completion marker.
                (out/'training_metadata.json').unlink()
                train.run_training(str(config),resume=True,smoke=True)
                restored=tiny_policy(cfg,str(out))
                for name,tensor in trained.state_dict().items():
                    torch.testing.assert_close(tensor,restored.state_dict()[name])
                sft=evaluate.run_evaluation(str(config),None,'sft',length=True,smoke=True)
                self.assertEqual(sft['sampled_kl_token_mean'],0.)
                self.assertEqual(sft['preference_accuracy'],0.)

if __name__=='__main__':unittest.main()

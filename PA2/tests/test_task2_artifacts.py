import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from task2_ppo.export_results import export

class PPOArtifactTests(unittest.TestCase):
    def test_export_checks_budgets_and_preserves_review(self):
        cfg={'results_dir':'results','prompts_per_update':1,'clip_values':[.05,.2,.5],'kl_values':[0.,.1,.2],
             'paths':{'rl_prompt_train':'train','rl_prompt_eval':'eval'}}
        train=[{'prompt_id':str(i)} for i in range(20)];evaluation=[{'prompt_id':'heldout'}]
        conditions={'standard':(20,.2,.1),**{f'clip_{e:.2f}':(8,e,.1) for e in cfg['clip_values']},**{f'kl_{b:.2f}':(8,.2,b) for b in cfg['kl_values']}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def artifact(p):return root/p
            def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
            for name in ['midpoint',*conditions]:
                folder=root/'results'/name
                write(folder/'evaluation_metadata.json',{'status':'complete','smoke':False})
                write(folder/'summary.json',{'condition':name,'reward_score_mean':1.})
                (folder/'generations.jsonl').write_text(json.dumps({'row_index':0,'prompt_id':'heldout','response':'text'})+'\n')
                if name!='midpoint':
                    n,e,b=conditions[name];out=root/'outputs/task2_ppo'/name
                    write(out/'training_metadata.json',{'status':'complete','smoke':False,'completed_updates':n,'clip_epsilon':e,'kl_beta':b,
                        'prompt_ids':[r['prompt_id'] for r in train[:n]],'generated_tokens':n*5,'wall_seconds':2.,'peak_vram_bytes':100})
                    write(out/'training_history.json',[{'update':i+1,'policy_gradient_norm':1.,'value_gradient_norm':1.,
                        'epochs':[{'policy_gradient_norm':1.,'value_gradient_norm':1.,'overflow_retries':0}]} for i in range(n)])
            write(root/'results/cached_clipping/diagnostics.json',{'smoke':False,'results':[{'epsilon':e} for e in cfg['clip_values']]})
            with patch('task2_ppo.export_results.load_yaml',return_value=cfg),patch('task2_ppo.export_results.artifact_path',side_effect=artifact), \
                 patch('task2_ppo.export_results.read_jsonl',side_effect=lambda p:train if p=='train' else evaluation):
                bad=root/'outputs/task2_ppo/standard/training_metadata.json';meta=json.loads(bad.read_text());meta['completed_updates']=19;write(bad,meta)
                with self.assertRaises(ValueError):export()
                meta['completed_updates']=20;write(bad,meta);export()
                review=root/'results/qualitative_review.jsonl';rows=[json.loads(l) for l in review.read_text().splitlines()]
                rows[0]['manual_quality_notes']='student note';review.write_text(''.join(json.dumps(r)+'\n' for r in rows));export()
                self.assertEqual(json.loads(review.read_text().splitlines()[0])['manual_quality_notes'],'student note')
                self.assertTrue(all(json.loads((root/'results/completion_checks.json').read_text()).values()))
if __name__=='__main__':unittest.main()

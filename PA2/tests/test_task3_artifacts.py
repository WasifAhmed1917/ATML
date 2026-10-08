import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from common.data import load_yaml
from task3_grpo.export_results import check_training


class GRPOArtifactTests(unittest.TestCase):
    def test_reject_smoke_wrong_budget_ids_and_unmatched_settings(self):
        cfg=load_yaml('configs/grpo.yaml');rows=[{'prompt_id':str(i)} for i in range(20)]
        meta={'status':'complete','smoke':False,'completed_updates':8,'loss_type':'grpo','num_generations':4,
              'prompt_ids':[r['prompt_id'] for r in rows[:8]],'midpoint_policy':cfg['paths']['grpo_midpoint_policy'],
              'max_generated_token_budget':8*4*512,'generated_tokens':100,'config':cfg}
        history=[{'update':i+1} for i in range(8)];check_training(meta,history,cfg,rows,8,'grpo')
        for change in ({'smoke':True},{'completed_updates':7},{'num_generations':2},{'prompt_ids':['wrong']},
                       {'max_generated_token_budget':4096},{'midpoint_policy':'wrong'},{'loss_type':'dr_grpo'}):
            with self.subTest(change=change),self.assertRaises(ValueError):check_training({**meta,**change},history,cfg,rows,8,'grpo')
        with self.assertRaises(ValueError):check_training({**meta,'config':{**cfg,'kl_beta':.2}},history,cfg,rows,8,'grpo')
        with self.assertRaises(ValueError):check_training(meta,history[::-1],cfg,rows,8,'grpo')

    def test_complete_export_and_all_figures_with_controlled_artifacts(self):
        from task3_grpo.export_results import export
        from task3_grpo.plot_results import plot
        from task3_grpo.runtime import write_json
        from task3_grpo.analyze_groups import analyze_cache
        cfg=load_yaml('configs/grpo.yaml');train=[{'prompt_id':str(i)} for i in range(20)]
        evaluation=[{'prompt_id':'heldout-'+str(i)} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'PA2_ARTIFACT_ROOT':tmp}), \
             patch('task3_grpo.export_results.read_jsonl',side_effect=lambda path:train if path==cfg['paths']['rl_prompt_train'] else evaluation), \
             patch('task3_grpo.export_results.file_hash',return_value='fixed-cache'):
            root=Path(tmp)/cfg['results_dir']
            for name in ('midpoint','standard','norm_grpo','norm_dr_grpo'):
                folder=root/name;write_json(folder/'evaluation_metadata.json',{'status':'complete','smoke':False,'config':cfg})
                write_json(folder/'summary.json',{'condition':name,'n_prompts':2,'reward_score_mean':1.,
                    'sampled_kl_token_mean':.01,'entropy_token_mean':.5,'response_length_mean':2.})
                records=[{'prompt_id':r['prompt_id'],'response':'example','reward_score':1.} for r in evaluation]
                (folder/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                if name=='midpoint':continue
                updates=20 if name=='standard' else 8;loss_type='dr_grpo' if name=='norm_dr_grpo' else 'grpo'
                actualcfg={**cfg,'updates':updates};out=Path(tmp)/'outputs/task3_grpo'/name
                meta={'status':'complete','smoke':False,'completed_updates':updates,'loss_type':loss_type,'num_generations':4,
                    'prompt_ids':[r['prompt_id'] for r in train[:updates]],'midpoint_policy':cfg['paths']['grpo_midpoint_policy'],
                    'max_generated_token_budget':updates*4*512,'generated_tokens':updates*8,'config':actualcfg,
                    'optimization_updates':updates,'wall_seconds':1.,'peak_vram_bytes':1024}
                write_json(out/'training_metadata.json',meta);history=[]
                for u in range(1,updates+1):
                    lengthstats={'completion_index':0,'response_length':2,'length_bin':'short_1_127','advantage':1.,
                        'policy_logp_abs_gradient_sum':.25,'policy_logp_abs_gradient_mean':.125,'normalization_denominator':2 if loss_type=='grpo' else 512}
                    history.append({'update':u,'generated_tokens':8,'reward':1.,'sampled_kl_log_ratio':.01,
                        'group_reward_std':1.,'uninformative_group_fraction':0.,'policy_term':0.,'gradient_norm':.1,
                        'entropy':.5,'response_length':2.,'truncated_fraction':0.,
                        'epochs':[{'policy_epoch':1,'gradient_norm':.1,'overflow_retries':0,'length_statistics':[lengthstats]}]})
                    write_json(out/'rollouts'/f'update_{u:03d}.json',[{'response_length':2,'truncated':False,'training_tokens':2}]*4)
                    (out/'rollouts'/f'update_{u:03d}.pt').write_bytes(b'test existence marker')
                write_json(out/'training_history.json',history)
            cache=[{'prompt_id':str(p),'source_index':p,'generation_index':i,'reward':float(p+i%2),
                    'clipped_at_max':False} for p in range(4) for i in range(8)]
            cached=analyze_cache(cache);cached['cache_sha256']='fixed-cache';write_json(root/'group_sizes'/'diagnostics.json',cached)
            import pandas as pd
            pd.DataFrame(cached['summary']).to_csv(root/'group_sizes'/'summary.csv',index=False)
            self.assertTrue(all(export().values()));plot()
            self.assertEqual(len(list((root/'figures').glob('*.png'))),4)
            self.assertTrue((root/'length_conditioned_summary.csv').exists())
            # Rerunning export must retain the student's manual review notes.
            review=root/'qualitative_review.jsonl';records=[json.loads(line) for line in review.read_text().splitlines()]
            records[0]['manual_quality_notes']='keep my note';review.write_text(''.join(json.dumps(r)+'\n' for r in records))
            export();self.assertEqual(json.loads(review.read_text().splitlines()[0])['manual_quality_notes'],'keep my note')

if __name__=='__main__':unittest.main()

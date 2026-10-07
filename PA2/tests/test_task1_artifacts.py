"""Artifact repair and result completeness checks using isolated fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from task1_dpo.utils import load_records, append_record
from task1_dpo.export_results import export_results


class ArtifactTests(unittest.TestCase):
    def test_interrupted_final_append_recovers_without_losing_records(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'records.jsonl'
            p.write_text('{"row_index":0}\n{"row_ind')
            self.assertEqual(load_records(p),[{'row_index':0}])
            append_record(p,{'row_index':1})
            self.assertEqual(load_records(p),[{'row_index':0},{'row_index':1}])
            p.write_text('{"row_index":0}')
            self.assertEqual(load_records(p),[{'row_index':0}])
            append_record(p,{'row_index':1})
            self.assertEqual(len(load_records(p)),2)
            p.write_text('{broken}\n{"row_index":1}\n')
            with self.assertRaises(json.JSONDecodeError):load_records(p)

    def test_export_rejects_missing_ids_then_preserves_manual_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);results=root/'results';results.mkdir()
            paths={}
            for key in ('dpo_standard_eval','dpo_length_eval','word_limit_prompts'):
                p=root/(key+'.jsonl')
                p.write_text(json.dumps({'prompt_id':'p0'})+'\n');paths[key]=str(p)
            config=root/'config.yaml';config.write_text(yaml.safe_dump({'results_dir':str(results),'paths':paths}))
            for name,n in {'standard':1500,'length_balanced':1500,'beta_0.03':600,'beta_0.10':600,'beta_0.30':600}.items():
                d=results/name;d.mkdir()
                summary={'condition':name,'smoke':False}
                if name in ('standard','length_balanced'):
                    summary['word_limit']={'n_prompts':1}
                    summary['strata']={s:{'n_pairs':1} for s in ('preferred_longer','length_matched','rejected_longer')}
                (d/'evaluation_summary.json').write_text(json.dumps(summary))
                (d/'run_metadata.json').write_text(json.dumps({'train_examples':n,'status':'complete'}))
                (d/'evaluation_metadata.json').write_text('{"status":"complete"}')
            with self.assertRaises(ValueError):export_results(str(config),require_complete=True)
            record={'prompt_id':'p0','row_index':0,'messages':[{'role':'user','content':'fixture'}],
                    'response':'fixture','reward_score':0.,'response_length':1,'sampled_kl_token_mean':0.,
                    'word_limit':None,'word_limit_compliant':None,'truncated':False}
            for d in results.iterdir():
                if not d.is_dir():continue
                for f in ('standard_pairs.jsonl','generations.jsonl','length_pairs.jsonl','word_limit_generations.jsonl'):
                    (d/f).write_text(json.dumps(record)+'\n')
            export_results(str(config),require_complete=True)
            self.assertTrue(all(json.loads((results/'completion_check.json').read_text()).values()))
            review=results/'qualitative_review.jsonl';rows=load_records(review)
            rows[0]['manual_quality_notes']='My own annotation.'
            review.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            export_results(str(config),require_complete=True)
            self.assertEqual(load_records(review)[0]['manual_quality_notes'],'My own annotation.')

if __name__=='__main__':unittest.main()

"""Download only Task 5 files from the same immutable course release."""
from huggingface_hub import snapshot_download
from common.data import repo_path
from task5_feedback.runtime import ASSET_REVISION


def main():
    snapshot_download(repo_id='AbDu11aHHH/ATML-PA2-assets',repo_type='dataset',revision=ASSET_REVISION,
                      local_dir=str(repo_path('.')),allow_patterns=['data/gsm8k_eval.jsonl','data/math_transfer_eval.jsonl',
                      'data/task5_controlled_reward_diagnostics.jsonl','checkpoints/rlvr_policy/**',
                      'checkpoints/rlaif_policy/**','manifests/**'])
    print('Task 5 fixed course assets downloaded; no dataset resampling or training.',flush=True)

if __name__=='__main__':main()

# Task 5: RLVR versus RLAIF

This is evaluation only. Use the untouched Qwen2.5-1.5B-Instruct SFT base and the two **supplied** `checkpoints/rlvr_policy` and `checkpoints/rlaif_policy` adapters. There is no dependency on the Task 4 human audit or the student-trained Task 3 policy.

The PA requirements are implemented in scripts, with the Colab notebook acting only as a launcher. Keep `requirements.txt`, `configs/base.yaml` and `configs/feedback.yaml` unchanged. `rlvr.py` and `rlaif.py` are copied unchanged from the released course starter (AbDu11aHHH/ATML-PA2-LLM-PostTraining); its policy/tokenizer/generation utilities are reused. Evaluation, recovery, metrics and exports are student implementation support.

From `PA2/`:

```bash
python -m pip install -r requirements.txt
python -m scripts.check_environment
python -m task5_feedback.download_assets
python -m task5_feedback.validate_data
python -m unittest discover -s tests -p 'test_task5*.py' -v
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm --resume
python -m task5_feedback.score_perturbations --config configs/feedback.yaml --resume
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer --resume
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
python -m task5_feedback.plot_results --config configs/feedback.yaml
```

Equivalent stage commands:

```bash
python -m task5_feedback.run_all --stage gsm --resume
python -m task5_feedback.run_all --stage diagnostics --resume
python -m task5_feedback.run_all --stage transfer --resume
python -m task5_feedback.run_all --stage export --resume
# Or all four, sequentially:
python -m task5_feedback.run_all --stage all --resume
```

## Fixed protocol

- Asset revision: `0b350481fb03f5525a35bcdec4131bd4fe487f98`. SHA-256 checks protect both supplied adapter weights/configurations and the fixed 300 GSM8K evaluation records, 100 SVAMP records and 100 controlled responses (20 problems × five variants). No filtering, resampling or editing.
- Use the released `messages` unchanged. GSM8K/SVAMP already request reasoning and a final `#### <number>`; no new CoT template is added.
- All policies: float16, frozen parameters, 512 new tokens, temperature 0.7, top-p 0.9 and sampling enabled, from `base.yaml`/`feedback.yaml`. The Task 5 scaffold leaves evaluation batch size unspecified: choose four and hold it fixed. Reset seed 6304 + original batch start for each policy's matching batch; this choice is recorded, not tuned using evaluation results. Retain full prompts and reject context overflow instead of truncating silently.
- Fixed pairwise judge: Qwen2.5-3B-Instruct; released CUDA 4-bit NF4 loading, original rubric, content-hash A/B balancing, greedy **four-token** generation, original parser with TIE fallback. The generic `judge_max_new_tokens: 64` in feedback.yaml applies to Task 4's safety judge; Task 5's released `PairwiseAIJudge.compare` hard-codes four. Both files/settings remain unchanged.
- Both RLVR and RLAIF are compared against matched SFT responses in each domain. Save raw win/tie/loss rates and the distinct tie-adjusted win rate. The SFT baseline has no fabricated self-judge win rate.
- Exact correctness uses the released **last designated final field**, ignoring gold numbers present only in intermediate reasoning. Formatting separately requires exactly one numeric final field on the last line. Length counts generated tokens through EOS, excludes padding, and includes mean/std/median/IQR.
- Controlled pairs: A=clean, B=each variant. For `clean_correct`, use the clean self-pair as an identity/tie control; a better response is undefined and blank in CSV. For other categories clean is the diagnostically better response; retain verifier ties as legitimate invariance. `S_reason` uses reasoning corruption; `S_outcome` uses good reasoning/wrong final. Filler and gold-distractor behavior are separate.

## Disconnects and evidence

Set `PA2_ARTIFACT_ROOT` to a persistent directory (the notebook uses `/content/drive/MyDrive/PA2`). Results are under `results/task5_feedback`; assets remain in the checkout. Completed rows flush immediately; incomplete final writes are repaired. A partially written generation batch replays with its original members and seed before appending missing rows. Replay checks already saved responses and refuses a mismatch. Code, configurations, source rows and adapter hashes guard resume. Keep the same GPU/software environment while resuming; do not pull changed Task 5 code midway through a run.

The supplied judge cache is reconstructed from durable pair records if its non-atomic write is interrupted; its labels and decision protocol are unchanged. A completed stage skips model loading. Resume may repeat the single uncommitted judge call or the partial generation batch, but does not lose earlier durable results.

`--smoke` on `generate_responses` and `judge_pairs` uses two math prompts with full settings. Use a separate artifact root; smoke output cannot satisfy full completion checks.

Full exports include:

- `summary.csv`, domain summary JSON: accuracy, formatting, length, truncation, pairwise scores and verifier–judge agreement with strict/tie denominators.
- `diagnostic_pairs.jsonl`, `diagnostic_rates.csv`, `sensitivity.csv`: all five categories, better/tie/wrong rates, S_reason/S_outcome, filler preference and distractor robustness.
- `transfer_drops.csv`, `failure_types.csv`: differences in percentage points and observable missing/wrong final failures, with truncation separately flagged. Different benchmark difficulty limits causal interpretation of a domain drop.
- `agreement_contingency.csv`, `policy_disagreements.json`, `qualitative_diagnostics.csv`: full examples spanning reasoning corruption, persuasive filler, wrong final and gold distractor. Selection is the first fixed-order disagreement per category, or first pair when none exists; manually explain the cases in the report.
- `fixed_inputs.json`, per-stage metadata and generated token IDs: fixed record IDs, dataset/adapter hashes, code/config/package/GPU provenance and original batch seeds.
- `inference_cost.json`, `comparison.json`, `completion_checks.json`, standalone PDF/PNG plots and `evidence_notes.md` for feedback coverage/noise/exploitability/cost discussion.

Expect 1,200 generated responses and 900 judge comparisons in a full run. Allow several hours on A100; actual duration depends on completion length. The live progress bars give stage estimates. GPU experiment results must be produced by running the notebook; local unit/mock tests do not constitute those results. No evaluation results are used to select training settings, and binary verifier rewards and AI win rates are not treated as calibrated utilities.

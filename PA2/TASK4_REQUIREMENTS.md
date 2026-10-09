# Task 4 — Safety calibration

Task 4 is post-hoc evaluation. Never use its labels to retune Tasks 1–3 or choose an ablation adapter.

| Requirement | Implementation / evidence |
| --- | --- |
| Four frozen policies | Untouched Qwen2.5-1.5B-Instruct SFT; completed standard DPO one epoch/1500 pairs/beta 0.1; standard PPO and GRPO 20 rollout updates. Preflight rejects smoke, incomplete runs, changed model families and substituted forks. |
| Common fixed benchmark | Pinned course XSTest CSV: 450 prompts, 250 SAFE / 200 UNSAFE, original row order. |
| Identical deterministic decoding | Released `generate_for_policy` protocol: greedy (`do_sample=False`), prompt cap 256, response cap 256, batch size 4, tokenizer right truncation. Generated token IDs, response text, length, EOS and truncation flags saved. |
| Fixed judge | Released Qwen2.5-3B-Instruct loader, categorical prompt, parser and `judge_one` unchanged. CUDA 4-bit NF4 with FP16 compute; deterministic judge cap 64. Confidence only for audit, never metric weighting. |
| Calibration rates | Safe-answer and over-refusal divided by all SAFE prompts; unsafe-compliance and justified-refusal divided by all UNSAFE prompts; ambiguous/all and ambiguous within classes. Ambiguous cases retained in denominators. Mean response tokens and truncation fractions included. |
| Category analysis | All five labels, zero counts included, for every policy × benchmark class × XSTest `type`. |
| Blind manual audit | Released NumPy `default_rng(6304)` selection: 30 SAFE plus 30 UNSAFE IDs, sorted. Same 60 IDs joined to each of four policies = 240 response-level labels. No AI label/confidence/rationale in blind sheet. Existing manual work preserved. |
| Manual agreement | Overall and per-policy exact agreement, full 5×5 confusion counts, AI/manual ambiguous rates. Partial labels explicitly marked pending; completion requires all 240. |
| Disagreements | Policy-label disagreements and human/AI disagreements exported with IDs; raw responses available for qualitative inspection. Student explains harmful compliance, justified refusal and over-refusal, and whether differences concern policy, judge or both. |
| Runtime reliability | Batch/row append persistence, atomic metadata; configuration/code/adapter/source identity guards. Interrupted generation replays original batch membership so padding width remains unchanged. Partial last JSONL append repair inherited from Task 1. |

The released selection balances SAFE/UNSAFE classes, not every semantic category equally. Preserve that fixed rule rather than resampling categories. The starter creates 60 prompt IDs and explicitly asks that they be joined to each policy; hence 240 manual response labels, not a substituted set of 60 policy-response pairs.

No changes were made to Task 1–3 code or configurations. `configs/feedback.yaml` is byte-identical to the starter. The judge prompt/parser/loader/single-example scorer and audit-ID selection function are retained verbatim. Judge loader defaults freeze the model operationally; the orchestration also disables parameter gradients. No reward model is used for categorical safety scoring.

## Run on Colab

Open `notebooks/ATML_PA2_T4.ipynb` after publishing the Task 4 files. Use the same Drive artifact root as Tasks 1–3, `MyDrive/PA2`.

1. Setup, install pinned requirements, download pinned assets and validate standard adapters.
2. Optional isolated two-prompt SFT smoke test.
3. Generate 450 responses per policy; create/download blind audit sheet.
4. Label the sheet without viewing AI labels. Judging can run while you label offline.
5. Run the fixed AI judge (1800 prompt-response pairs).
6. Upload the completed sheet, export rates/agreements/plots, download results.

After disconnect: rerun setup/dependencies, then resume unfinished generation or judging stage with `--resume`. Completed rows are skipped. Keep source and batch size fixed when resuming. Export may run before manual labeling; it reports `manual_audit_complete: false` until human labels are supplied.

CLI: `python -m task4_safety.run_all --stage generate --resume`, then `--stage audit`, `--stage judge --resume`, and `--stage export`. `all` also creates the blind sheet before judging but does not invent or fill manual labels.

On an A100, budget approximately 2–4 hours for full generation and AI judging; actual length, model loading and 4-bit decoding speed determine runtime. Manual audit time is additional. This is a planning estimate, not a measured full GPU run.

## Required student analysis

Report all calibration rates, category distributions and length/truncation information; judge/manual agreement and confusion; selected harmful compliance, justified refusal and over-refusal examples. Discuss whether learned reward improvements in Tasks 1–3 correspond to calibration, and qualify observational policy differences with judge errors. Human labeling and the final interpretation are not automated.

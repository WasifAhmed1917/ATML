# Task 3 implementation and release settings

Source: Task 3 in the assignment manual; pinned starter `configs/grpo.yaml`, `configs/base.yaml`, and `task3_grpo/grpo.py`.

| Requirement | Implementation |
|---|---|
| Supplied GRPO midpoint | Exact `checkpoints/grpo_midpoint_policy` adapter for standard and both forks; fresh optimizer as released |
| Objective defect | `group_relative_advantages`: mean and population standard deviation separately within each prompt group, never across unrelated prompts |
| Standard continuation | 20 rollout updates, one prompt/update, K=4 completions/prompt |
| Generation and loss mask | Prompt cap 256, completion cap 512; entire max-length non-EOS completion excluded from policy and KL losses |
| Group-size diagnostic | Same 192 cached generations, 24 prompts × eight completions; consecutive disjoint partitions for K=2/4/8 |
| Equal generation budgets | 192 generations at each K; 96 per difficulty bin; respectively 96/48/24 groups overall |
| Fixed difficulty bins | Rank 24 prompts once by mean learned reward across their full eight cached completions, tie-break by prompt ID; split into two sets of 12 |
| Matched normalization forks | Eight updates each, K=4, identical midpoint, first eight training prompts, per-update seeds, reward, optimizer, epsilon, beta, generation caps/settings |
| Canonical normalization | Policy surrogate divided by each completion's valid token count, then averaged over all sampled completions |
| Supplied Dr.GRPO-style normalization | Policy surrogate divided by constant max completion length 512, then averaged over all sampled completions |
| Reward/reference | Released frozen reward model; original base policy with LoRA disabled for reference |
| Optimizer and precision | AdamW 5e-6, released default weight decay 0.01; frozen base FP16, trainable LoRA FP32; max gradient norm 1 |
| Other settings | Seed 6304; one policy epoch/update; epsilon 0.20, beta 0.10; temperature 0.7, top-p 0.9 |
| Held-out protocol | Same 200 prompt IDs as PPO; prompt cap 256, evaluation completion cap 768, reward input cap 1280 |
| Diagnostics | Reward, sampled reference log-ratio KL, objective k3 KL, group reward std/uninformative fraction, policy loss, gradient norm, exact categorical entropy, length, clipping, truncation, eligible tokens, peak VRAM, wall time |
| Length-conditioned evidence | Exact absolute policy-surrogate gradient with respect to sampled token logp, separately by completion and fixed length bins 1–127/128–255/256+; excludes KL |
| Recovery | Atomic checkpoint after each update: trainable policy, optimizer, scaler, RNG, histories; per-response evaluation resume and code/config/ID checks |
| Saved evidence | All rollout tensors and prompt/message/response/token IDs/reward/advantage/mask records; all held-out generations; CSVs and plots; blank qualitative-review fields |

The supplied Dr.GRPO-style condition changes only the policy-surrogate denominator. Both conditions retain within-group reward-standard-deviation normalization and the starter's globally token-normalized k3 KL term. This is the course's controlled variant; it does not implement every modification discussed in the Dr.GRPO paper.

The cache study retains all eight completions per prompt at every K, partitions by generation index, and computes advantages independently within each partition. A group is informative when its population reward standard deviation exceeds 1e-6. Difficulty bins are a learned-reward proxy, not ground-truth problem difficulty. Signal variance means within-group variance of normalized advantages, averaged across groups; it is near one for informative groups by construction. Raw centered-reward variance and post-mask effective signal are also exported to make that limitation visible. Cache masking uses the supplied 768-token truncation flags; fresh training masking uses its 512-token cap.

Training advantages include all sampled rewards before applying the completion-level truncation mask, following the released helpers. Masked completions remain in the sequence-average denominator and contribute zero. If every completion is truncated, the rollout update is logged but no optimizer step is taken, including no AdamW decay; the number of actual optimization updates is reported separately. The isolated smoke run uses a 16-token cap and disables truncation masking **only for that test** so that FP16 backward is exercised. Full runs require masking enabled, and exports reject smoke artifacts.

Each eight-update fork has the same maximum generation budget, 8 × 4 × 512 = 16,384 tokens. Natural EOS stopping means realized generated-token totals may differ; both totals are exported. No extra rollout generation is performed to force equality, because that would change prompts or updates.

Prompt truncation retains the released right-truncation convention, consistently with Task 2, and evaluation logs its frequency. Dropout is disabled during generation and replay, with gradients enabled during replay. Raw teacher-forced model log probabilities define likelihood ratios and KL; sampling uses the released temperature/top-p settings. KL log-ratio estimates can be negative on finite samples; the separately named k3 statistic matches the training penalty. The reference is the original base model, not the supplied midpoint.

The evaluation-only 768/1280 caps match Task 2's held-out and reward protocol; they do not change released GRPO training caps. GPU wall time covers rollout collection, optimization and checkpointing, excluding model loading and held-out evaluation. Numeric exports and figures are automatic; report analysis and qualitative judgments remain the student's work.

Development validation uses real tiny Qwen2 + PEFT optimization, full-batch versus per-completion gradient equivalence for both normalizations, checkpoint/optimizer recovery, truncation/no-decay cases, prompt-independent advantages, and equal cached generation budgets. Full-size GPU smoke and experiments must run in Colab; development results do not claim course-model training completion.

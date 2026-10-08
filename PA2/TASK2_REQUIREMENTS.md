# Task 2 implementation and release settings

Source: Task 2 in the assignment manual and pinned starter `configs/ppo.yaml` / `configs/base.yaml`.

| Requirement | Implementation |
|---|---|
| Supplied PPO midpoint policy and matched critic | `continue_train.prepare_ppo_continuation`; same assets for all branches |
| Correct clipped surrogate | `ppo.ppo_policy_loss`: minimum of unclipped and clipped terms, sign-aware tests |
| 20-update standard continuation | `run_all --stage standard` |
| 8-update clipping forks, epsilon 0.05/0.20/0.50 | `run_all --stage clipping` |
| Cached rollout geometry | Stored rewards/values/old/reference logp; current midpoint replay; all 32 rows; exact token-count checks |
| 8-update KL forks, beta 0/0.10/0.20 | `run_all --stage kl` |
| Fixed prompts, seeds, optimization and nominal token budgets | Identical first eight supplied training rows for all forks; per-update seed 6304 + update; same generation caps |
| Reward and reference | Released frozen 8-bit reward model; base policy with LoRA disabled for reference |
| Critic continuation | Released merged 0.5B critic; fresh zero-output rank-8 LoRA and trainable scalar head |
| Training/evaluation caps | Prompt 256; training response 512; evaluation response 768; reward input 1280 |
| Optimizers | Policy AdamW 3e-6 with released default weight decay 0.01; critic LoRA 1e-4/head 3e-4, weight decay 0 |
| Other settings | Two PPO epochs/update, one prompt/update, gamma 1, lambda 0.95, value coefficient 0.5, missing-EOS penalty 1, max grad norm 1 |
| Diagnostics | Reward/raw and effective, reference KL, policy/value loss, exact categorical entropy, clip/active fractions, both gradient norms, approximate old-policy KL, EOS/length, elapsed time and peak VRAM |
| Reproducibility and recovery | Fixed IDs and configuration/code signatures; atomic per-update checkpoint with trainable parameters, optimizers, scaler, RNG and history |
| Examples and rollouts | Per-update JSON and tensor bundle; every held-out prompt/response/token IDs/reward/KL/entropy/termination saved |
| Required evidence | Summary CSV, trajectories CSV, cached diagnostics CSV, four numeric figures, qualitative review JSONL |

Prompt tokenization keeps the released tokenizer's right-truncation convention consistently in training, cache reconstruction and evaluation; prompt truncation is logged. Frozen model bases retain FP16; trainable LoRA/head parameters use FP32 for safe GradScaler updates. PPO replay uses evaluation mode with gradients enabled, disabling dropout so old/new probabilities refer to the same deterministic network. Probabilities and entropy use the raw model logits, following the released teacher-forced helpers; generation uses temperature 0.7 and top-p 0.9. KL is the sampled log-probability difference, not an exact nonnegative KL computation. The critic scores states immediately before each response action; GAE uses zero bootstrap at the final valid token. Advantages normalize over valid response tokens; the cached study normalizes once across its complete fixed batch.

The standard run and short forks have different update budgets. All short forks share a maximum 4096 generated-token budget (8 x 512); EOS may reduce realized tokens, which are exported separately. GPU training wall time excludes loading models and held-out evaluation and includes rollout collection, optimization and checkpoint writes. The supplied critic's documented weak generalization is retained without offline repair.

Validation in the development environment uses objective tests and real tiny Qwen2 + PEFT actor/critic integration, including checkpoint/optimizer recovery and joint FP16 overflow replay. Full-size GPU smoke and course experiments must be run in Colab; no GPU results are claimed here. Manual report discussion and qualitative judgments remain the student's work.

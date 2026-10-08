# ATML PA2 — Task 1: DPO

Task 1 implementation built on the course starter. Only Task 1 and shared
infrastructure are included. No GPU experiment results or report analysis are
supplied. Run the notebook to produce your results.

[Open ATML_PA2_T1.ipynb in Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA2/notebooks/ATML_PA2_T1.ipynb)

## Colab

Open the Colab link above. `CODE_SOURCE = "github"` is the default and clones
this repository. Optional ZIP mode is available by setting `CODE_SOURCE = "zip"`
and uploading a package containing the `PA2/` directory.

Choose an A100 GPU runtime. Run the notebook cells in order: mount Drive, clone
this repository, install the released requirements, download and validate the
pinned course assets, validate Task 1 data, run CPU tests, run a small real-model
GPU smoke test, then execute standard, beta, and balanced stages.

The notebook sets `PA2_ARTIFACT_ROOT=/content/drive/MyDrive/PA2`. Adapters,
recoverable training checkpoints, logs, and per-example evaluations are written
there directly. Models and downloaded course assets stay on Colab's local disk.
Rerun setup in a fresh session and use the same Drive folder to resume.
Each script reads its inputs from disk; no training logic lives in the notebook.

## Structure

```text
PA2/
  common/                 shared course model/data/metric helpers
  configs/                original released settings
  scripts/                course downloader and asset validator
  task1_dpo/              objective, training, evaluation, exports, plots
  tests/                  CPU objective/preprocessing/integration checks
  notebooks/ATML_PA2_T1.ipynb
  data/                   downloaded fixed course data; ignored
  cached/                 downloaded course diagnostics; ignored
  checkpoints/            supplied course checkpoints; ignored
  outputs/task1_dpo/      trained adapters and resume state; ignored
  results/task1_dpo/      numeric results, generations, provenance, figures
```

`PA2_ARTIFACT_ROOT` redirects relative `outputs/` and `results/` paths to the
persistent folder. Without it, these folders live under `PA2/` locally.
Keep small results in GitHub after reviewing them. Do not commit public model
weights, raw course datasets, model caches, or training checkpoints.

## Fixed release and experiments

- Starter: https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining
- Starter snapshot used: `1d64ac6` (as recorded in the supplied archive).
- Assets: https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets
- Asset revision: `0b350481fb03f5525a35bcdec4131bd4fe487f98`.
- Policy/reference: `Qwen/Qwen2.5-1.5B-Instruct`; reference is the frozen original
  base with LoRA disabled. Every run initializes fresh LoRA from the same seed.
- Reward: `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, frozen and loaded with the
  released 8-bit setting; canonical Qwen tokenizer as required by the starter.
- Seed 6304, FP16 base, LoRA rank 8/alpha 16/dropout .05 on `q_proj`/`v_proj`.
- AdamW, learning rate 2e-5, weight decay 0, batch 2, accumulation 8, clipping 1.
- Maximum combined pair sequence 768; generated responses capped at 256 tokens;
  sampling temperature .7 and top-p .9. No extra CoT instruction is inserted.

| Condition | Training pairs | Epochs | Beta |
|---|---:|---:|---:|
| standard | 1,500 | 1 | .10 |
| beta_0.03 | first 600 standard pairs | 1 | .03 |
| beta_0.10 | same 600 pairs | 1 | .10 |
| beta_0.30 | same 600 pairs | 1 | .30 |
| length_balanced | supplied 1,500 pairs; 500 per stratum | 1 | .10 |

Held-out preference evaluation and generation use the fixed 300 standard pairs.
Standard and balanced runs also evaluate the fixed 246 stratified pairs (82 per
stratum) and the 10 common word-limit prompts. Optional SFT evaluation supplies
an untrained baseline for manual response review; its DPO margin is identically
zero, so strict preference accuracy is zero, not a useful SFT quality metric.
The three 600-pair runs are separate from the 1,500-pair standard baseline.

## Preprocessing decision

The supplied archive states that the TA permits several truncation approaches.
That permission is not documented in the attached assignment; confirm it with
the course staff before treating the preprocessing choice as approved.
We use **response-prioritizing truncation** for all DPO preference training and
evaluation. No row is filtered and no supplied index or stratum label is changed.

`task1_dpo/encoding.py` renders Qwen's chat template and appends EOS to each
response. It budgets prompt space using the **longer** chosen/rejected response,
then keeps the same prompt tail for both sequences. This is a pair-consistent
adaptation of the initial starter's prompt-tail truncation, which budgeted each
response independently. If a response itself exceeds the remaining space, its
tail is kept, including EOS, with at least one prompt token retained. This
case can lose response content and is explicitly logged. The data audit found
98/1,500 standard-training pairs and 0/1,500 balanced-training pairs with
response truncation under this rule.

Both sequences stay within 768 tokens. Only response tokens, including EOS, are
scored; prompt and padding targets are masked out. Original/retained/removed
lengths are saved per pair and summarized per file and stratum. Truncation can
remove necessary context or response content; inspect these diagnostics when
writing your own length-study analysis.

Generation receives only the prompt and uses left truncation above 768 prompt
tokens; this is separate from pair preprocessing. The original messages and
prompt-truncation flag are saved. Reward scoring uses the course helper's
1024-token limit, with reward-input truncation logged. Generation cut-offs at
256 tokens are recorded separately from all input truncation.

## Reproduction commands

Run from `PA2/`:

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m task1_dpo.validate_data --tokenize
python -m unittest discover -s tests -v

# Separate processes release GPU memory between stages.
python -m task1_dpo.run_all --stage standard --resume
python -m task1_dpo.run_all --stage beta --resume
python -m task1_dpo.run_all --stage length --resume
python -m task1_dpo.run_all --stage export --resume
python -m task1_dpo.plot_results
```

Or use `python -m task1_dpo.run_all --resume` for the entire pipeline.
`--resume` resumes a partial run and skips matching completed training. It fails
if code, preprocessing, config, or data differ, preventing mixed experiments.
Evaluate standard with `--length` from its first evaluation so its protocol
matches the later balanced comparison. The notebook does this automatically.

Individual entry points:

```bash
python -m task1_dpo.train --run-name standard --resume --skip-completed
python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard --length --resume
python -m task1_dpo.ablate_beta --resume
python -m task1_dpo.analyze_length --resume
python -m task1_dpo.export_results --require-complete
```

The separate smoke run is a few training/evaluation examples with the actual
model, reward model, and course caps. It is flagged `smoke=true` and excluded
from full-result exports. Its output path must be separate from standard runs.

## Results

Each condition saves:

- `run_metadata.json`, `training_ids.json`, `training_log.jsonl`/`.csv`.
- `evaluation_metadata.json`, `evaluation_summary.json`.
- `standard_pairs.jsonl`: response log-probabilities, preference margin/loss,
  accuracy and truncation diagnostics.
- `generations_raw.jsonl`/`generations.jsonl`: exact IDs, original messages,
  response text/token IDs, sampled KL, lengths, reward, EOS and cut-off flags.
- For standard/balanced: `length_pairs.jsonl` and word-limit generations.

Top-level files include `summary.csv`, `length_strata.csv`,
`word_limit_summary.csv`, `fixed_data_manifest.json`, exact `beta_subset_ids.json`,
`preprocessing_diagnostics.json`/`.csv`, and `completion_check.json`.
`qualitative_review.jsonl` aligns responses across conditions for your manual
correctness/concision/instruction review and retains your annotations on export.
The scripts do not claim automatic correctness judgments or write report prose.
`plot_results` regenerates numeric plots solely from the saved CSV files.

KL uses the released sampled log-probability-difference helper, summed over all
valid generated response tokens and divided by the total token count (EOS
included). It is a sampled diagnostic, not exact distributional KL. Reward
scores are uncalibrated scalar scores. Length SD uses population SD, with median
and IQR also saved. Preference accuracy uses strict reference-adjusted margin
`> 0`, with ties logged. Seeds are reset per generation prompt to support resume
and matched comparisons. Training records wall time and peak allocated VRAM.

## Validation and attribution

The DPO sign defect is corrected to `beta * (policy_margin - reference_margin)`.
CPU tests cover objective gradients, next-token/response masking, shared prompt
truncation, unchanged frozen reference, real tiny-Qwen/PEFT training, save/load,
checkpoint recovery, and evaluation resume. These checks use fixtures, not
course performance results. Full-model A100 validation occurs in the notebook.

Shared infrastructure, configurations and fixed prompts are materially reused
from the course starter linked above. Task 1
implementation and Colab launcher were developed with ChatGPT/Codex coding
assistance. The student must understand and validate submitted code. The PA
prohibits AI-written report language, interpretation, and analysis; those remain
the student's responsibility.

DPO objective: Rafailov et al., *Direct Preference Optimization: Your Language
Model is Secretly a Reward Model* (2023), https://arxiv.org/abs/2305.18290.

## Task 1 requirement audit

See [TASK1_REQUIREMENTS.md](TASK1_REQUIREMENTS.md) for the implementation mapping,
validation scope, and remaining experimental evidence. This upload is code-ready,
not a completed experimental submission.

## Task 2 — PPO continuation

Task 2 is implemented independently of the completed Task 1. Open
[the Task 2 Colab notebook](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA2/notebooks/ATML_PA2_T2.ipynb)
in an A100 GPU runtime. Its helper streams tqdm and saves logs on Drive.

Run setup, asset validation, objective tests and the isolated GPU smoke test, then
`standard`, `clipping`, `kl`, and `export` in order. Every stage accepts `--resume`;
completed training is skipped and evaluation resumes after the last saved response.
After a runtime reset, restore setup and assets before resuming an unfinished stage.
Keep the same code/configuration for partial runs. Do not run two copies of the same
stage concurrently.

```bash
python -m task2_ppo.run_all --stage standard --resume
python -m task2_ppo.run_all --stage clipping --resume
python -m task2_ppo.run_all --stage kl --resume
python -m task2_ppo.run_all --stage export --resume
```

See [TASK2_REQUIREMENTS.md](TASK2_REQUIREMENTS.md) for exact release settings,
probability/advantage conventions, and validation scope. Artifacts use
`PA2_ARTIFACT_ROOT` (the notebook sets `/content/drive/MyDrive/PA2`). The standard
adapter is at `outputs/task2_ppo/standard`; the critic continuation adapter is its
`critic_adapter` subdirectory. Tensor checkpoints retain the exact merged critic
origin rather than requiring a new critic initialization. Results are under
`results/task2_ppo`: fixed held-out generations, numeric summaries, cached clipping
geometry, trajectories, figures and `qualitative_review.jsonl` for manual review.
Task 2 reads the course assets; it does not modify Task 1 results or code.


## Task 3: GRPO

Open `notebooks/ATML_PA2_T3.ipynb` in Colab with an A100 GPU and run cells in order. It resumes per-update training and per-prompt evaluation from `MyDrive/PA2`, streams live progress, and keeps smoke artifacts separate. The standard 20-update continuation, equal-generation cached K study, and two matched eight-update normalization forks are implemented.

```bash
python -m task3_grpo.validate_data
python -m unittest discover -s tests -p 'test_task3*.py' -v
python -m task3_grpo.run_all --stage all --resume
```

Read `TASK3_REQUIREMENTS.md` for the objective fix, exact configuration, cache regrouping/difficulty-bin rule, truncation mask, and the scope of the supplied Dr.GRPO-style variant. Code and local integration checks are complete; GPU experiment results are produced when you run the notebook.

# Task 1 requirement audit

Scope: Task 1 and the common reproducibility/deliverable requirements in the
provided ATML PA2 assignment (pages 1–5 and 14–15). Tasks 2–5 scaffolds from the source archive are excluded from this upload.

**Status: implementation supplied; required experiment evidence is pending.**
No full-model GPU runs, performance measurements, qualitative conclusions, or
report are included. Software fixture tests do not establish model performance.

| Requirement | Implementation / evidence location | Status |
| --- | --- | --- |
| Correct reference-adjusted DPO loss; response-only summed log probabilities | `task1_dpo/dpo.py`, `utils.py`, objective tests | Implemented; CPU test coverage supplied |
| Standard fixed-set run, fresh Qwen LoRA, one epoch | `train.py`, `configs/base.yaml`, `configs/dpo.yaml` | Implemented; full run pending |
| Beta .03/.10/.30 forks, fresh initialization and identical 600-pair subset/settings | `run_all.py`, `ablate_beta.py`, `results/task1_dpo/beta_subset_ids.json` | Implemented; full runs pending |
| Supplied balanced data, unchanged strata, fresh initialization | `analyze_length.py`, `run_all.py`, `validate_data.py` | Implemented; full run pending |
| Held-out loss/accuracy, sampled token-mean KL, reward, mean length and dispersion | `evaluate.py`, `common/metrics.py` | Implemented; measured results pending |
| Standard/balanced accuracy in all three length strata | `evaluate.py` (`--length`) | Implemented; measured results pending |
| Common-prompt generated length and word-limit compliance | `evaluate.py`, tracked fixed word-limit prompts | Implemented; measured results pending |
| Distinguish full-run and short-fork budgets | README, run metadata, exported summary | Implemented |
| Fixed seeds, IDs, configurations, logs and decoding | `train.py`, `evaluate.py`, `utils.py`, `validate_data.py` | Implemented; actual run logs pending |
| Qualitative reward/quality disagreement and length/instruction examples | `export_results.py` produces `qualitative_review.jsonl` | Actual responses and manual review pending |
| Reproducible scripts, exact commands and attribution | `README.md`, notebook launcher, task modules | Supplied |
| Public GitHub; exclude raw datasets and weights | PA2 ignore rules and publication file review | Checked for this upload |
| Report with required evidence and repository link | Student-authored report | Not supplied; outside this code upload |

## Verification limits

- `results/task1_dpo/software_validation.json` and the fixed-data manifest came
  from the ZIP. They are historical claims, not evidence of a new run here.
- The separate `publication_validation.json` records checks performed before
  this GitHub upload, including the precise environment and test outcomes.
- The archive does not include downloaded course datasets or full-model weights.
  Re-run the asset and tokenization validators in Colab before training.
- The archive claims TA approval for response-prioritizing truncation, but the
  supplied assignment does not document that permission. The implementation
  consistently logs prompt/response truncation; course approval remains unverified.
- CPU fixtures cannot validate CUDA, FP16, 8-bit reward loading, A100 memory use,
  or the full Qwen/reward checkpoints. Run notebook section 5 before full runs.

## Remaining steps

1. Open the Task 1 Colab notebook from the README using an A100 runtime.
2. Run setup, pinned asset validation, tokenization validation, CPU tests and the
   real-model GPU smoke test.
3. Run standard, beta and length stages, then export and plot. The export uses
   `--require-complete` to check required conditions, budgets and saved IDs.
4. Review actual generations, select the required qualitative cases and write
   your own interpretation/report.
5. Copy reviewed small results and logs from Drive into `results/task1_dpo/`
   and commit them. Keep adapters, raw course data and public weights out of Git.

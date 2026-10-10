# Local validation — 10 October 2026

13 tests pass with the released Transformers 4.57.1, Tokenizers 0.22.1, PEFT 0.17.1 and TRL 0.27.2. Local PyTorch is 2.14.1+cpu (within the released torch>=2.4 requirement). No GPU is available in this implementation workspace.

Validated behavior: last designated final versus gold-number distractors; independent formatting/correctness; strict and tie agreement denominators; partial JSONL recovery; changed-input/config refusal; interrupted judge-cache reconstruction; completed judge-row skipping; original hash-balanced A/B mapping and four-token cap; partial generation-batch replay; exclusion of smoke results from full export; all fixed-set tables, qualitative pairs, PNG/PDF figures and completion checks; actual generation and response masks through a small randomly initialized Qwen2 model using the shared generation helper.

Full aggregation tests use synthetic responses/labels in temporary directories. They are software tests, not experiment evidence, and no synthetic metrics are included in the package. All 300 GSM8K, 100 SVAMP and 100 diagnostic source rows, and both supplied adapter weights/configuration files, pass pinned-release SHA-256 checks. The unchanged course verifier produces S_reason=0, S_outcome=1 and distractor robustness=1 on the fixed diagnostics; AI-judge values still require actual model evaluation.

`requirements.txt`, `configs/base.yaml`, `configs/feedback.yaml`, `rlvr.py` and `rlaif.py` match the released files byte for byte. Source files and notebook code cells parse successfully. Full SFT/RLVR/RLAIF GPU generation and judging must still be run in Colab to produce the PA's empirical evidence.

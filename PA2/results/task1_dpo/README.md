# Task 1 result files

The committed files currently contain **data/preprocessing validation only**.
No full-model training or evaluation has been run in this workspace.

- `fixed_data_manifest.json`: verified course-file hashes, counts, and truncation summaries.
- `beta_subset_ids.json`: exact first-600 pair IDs used for every beta run.
- The notebook generates `preprocessing_diagnostics.json` / `.csv`: original/retained
  token counts and flags, without raw prompts or preference responses.
- `software_validation.json`: CPU fixture tests and notebook validation status.

The Colab notebook creates numeric experiment summaries, logs, generated
responses, adapters, and plots in the persistent Drive folder. Copy reviewed
small results here for submission after the experiments finish. Do not present
fixture outputs or preprocessing counts as model-performance measurements.

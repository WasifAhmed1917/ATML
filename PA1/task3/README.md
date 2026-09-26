# Task 3: Domain generalization

Complete Task 2 first, then run `ATML_PA1_Task3.ipynb` in a Colab GPU runtime. The notebook reuses the Task 2 source splits and unchanged source-only checkpoint as ERM; additional methods are DAN-DG and SAM.

The setup requires these files under `/content/drive/MyDrive/ATML_PA1/Task2_PACS_stabilized/results/`:

- `imagenet_initial.pt`, `source_only_erm.pt`, `source_only.pt`
- `splits/pacs_sketch_seed6304.json`
- `task3_settings_locked_before_target_labels.json`
- `config.json`, `task2_summary.csv`

It checks matching checkpoint hashes and locked settings. PACS is expected under `/content/PACS_extracted/PACS`; the download cell recreates that layout if absent. Sketch is reserved for final evaluation, after source-only training and diagnostics. Do not adjust the locked settings based on Task 2 target performance.

Output: `/content/drive/MyDrive/ATML_PA1/Task3_PACS/results`. Export configuration, `task3_summary.csv`, source diagnostics, Sketch per-class/confusion/failure tables, strength study, and the Task 2 DAN versus Task 3 DAN-DG comparison to `results/`.

See the [PA1 README](../README.md) for dependencies, attribution, and repository status.

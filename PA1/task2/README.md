# Task 2: Unsupervised domain adaptation

Run `ATML_PA1_Task2.ipynb` in a Colab GPU runtime from setup through final evaluation. The notebook downloads/verifies PACS and uses Photo, Art Painting, and Cartoon as labeled sources with Sketch as the unlabeled adaptation domain. It trains source-only, DAN strengths 0.1/1/10, DANN, and CDAN.

Output: `/content/drive/MyDrive/ATML_PA1/Task2_PACS_stabilized/results`.

Checkpoint selection uses source-validation macro-F1. Preserve the Task 3 settings lock produced before final labeled Sketch evaluation. Keep `imagenet_initial.pt`, `source_only.pt`, and `source_only_erm.pt` in Drive for Task 3, along with `config.json`, `splits/pacs_sketch_seed6304.json`, `task3_settings_locked_before_target_labels.json`, and `task2_summary.csv`.

Copy small CSV/JSON exports and source split indices into `results/`. Keep checkpoints outside Git. Target labels must not be used to select models or settings.

See the [PA1 README](../README.md) for dependencies, attribution, and repository status.

## Python implementation and results

The Python stages and `scripts/run_task2.py` entry point are documented in [PYTHON_MODULES.md](../PYTHON_MODULES.md). [CSV/JSON results and supporting artifacts](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing) are linked on Drive.

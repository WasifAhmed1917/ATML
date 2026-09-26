# Task 4: Open-set recognition

Run `ATML_PA1_Task4.ipynb` in a Colab GPU runtime, in cell order. It downloads CIFAR data, constructs the fixed known split, trains Vanilla and GCSC, and initializes PROSER from the selected Vanilla checkpoint. It evaluates MSP, MLS, Energy, and diagonal Mahalanobis scores, plus PROSER's placeholder score.

Output root: `/content/drive/MyDrive/ATML_PA1/Task4_CIFAR_OSR`, with `results/`, `arrays/`, and `checkpoints/` subdirectories.

CIFAR-100 near/far unknown groups are evaluation-only. Thresholds use the 95th percentile of known CIFAR-10 validation unknownness. Keep checkpoints and cached features/logits in Drive. Export the split/unknown identifiers, `frozen_protocol.json`, `environment.json`, `vanilla_scores.csv`, `trained_models.csv`, per-class acceptance, and failure/score-disagreement tables. Copy small CSV/JSON exports into `results/`; the default Git ignore excludes large NumPy arrays, so any small split-only NPZ would need deliberate inclusion or conversion to JSON.

See the [PA1 README](../README.md) for dependencies, attribution, and repository status.

## Python implementation and results

The Python stages and `scripts/run_task4.py` entry point are documented in [PYTHON_MODULES.md](../PYTHON_MODULES.md). [CSV/JSON results and supporting artifacts](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing) are linked on Drive.

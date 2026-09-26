# Task 1: Inductive biases and representations

Run either notebook in a Colab GPU runtime, cell by cell. Both compare frozen ResNet-50, ViT-B/16, and CLIP on STL-10 with seed 6304.

- `ATML_PA1_Task1.ipynb`: AdaIN cue conflicts; output `/content/task1_results`. Run the final Drive backup cell before ending the runtime.
- `ATML_PA1_Task1_Controlled.ipynb`: VGG19 iterative style transfer with OpenCV foreground masks; output `/content/drive/MyDrive/ATML_PA1/Task1_STL10_Controlled`.

At the cue-conflict review widget, accept/reject images visually before prediction-based evaluation. Run candidate-expansion cells only when additional candidates are needed; they are conditional workflow steps. Continue after the required balanced accepted set exists. Preserve review decisions and do not combine results from the two variants without identifying them.

Export `config.json`, training/validation IDs, selected test IDs, cue-review and accepted-ID CSVs, `performance.csv`, `shape_bias.csv`, translation CSVs, and representation-stability/projection outputs. Keep AdaIN and controlled exports in separate subfolders of `results/`.

See the [PA1 README](../README.md) for dependencies, attribution, and repository status.

## Python implementation and results

The Python stages and `scripts/run_task1.py` entry point are documented in [PYTHON_MODULES.md](../PYTHON_MODULES.md). [CSV/JSON results and supporting artifacts](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing) are linked on Drive.

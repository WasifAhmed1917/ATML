# Python stages and reproduction

The `.py` files contain the actual code from the submitted notebooks, organized into setup/configuration, data preparation, models, training, and evaluation stages. They are not placeholder modules or a second implementation of the experiments. Each file identifies its original notebook and zero-based cell indices. Each `configs/pipeline.json` records stage order, source indices, and SHA-256 hashes.

These are **stage scripts with shared state**, not standalone importable APIs. The runner executes them in one namespace, preserving notebook variables, function globals, and execution order. Importing individual stage files is unsupported and may start downloads or training. No hyperparameters, model objectives, or output paths were changed. Dependency-install notebook magic is omitted; install `requirements.txt` first. Original notebooks remain unchanged.

## Tasks 2–4

In Colab, clone this repository to `/content/ATML`, enable a GPU, install missing requirements, and mount Drive in the notebook session. Run the Python entry point with `%run` so the original Drive and display behavior remains in the Colab kernel:

```python
%run /content/ATML/PA1/task2/scripts/run_task2.py
# Complete Task 2 before Task 3, retaining its Drive artifacts.
%run /content/ATML/PA1/task3/scripts/run_task3.py
%run /content/ATML/PA1/task4/scripts/run_task4.py
```

These commands perform full experiments, including training if checkpoints do not already satisfy the original notebook's resume logic. Preserve the original Task 2 lock-before-target-evaluation order. Task 3 reuses the Task 2 source splits and ERM checkpoint. This packaging does not convert the Colab-specific code into a portable CPU/desktop application.

Inspect execution order without ML imports, data downloads, or training:

```bash
python PA1/task2/scripts/run_task2.py --list
python PA1/task3/scripts/run_task3.py --list
python PA1/task4/scripts/run_task4.py --list
```

## Task 1: interactive review

Task 1 requires the cue-review widget in a live notebook session. Start the staged Python implementation in a Colab cell:

```python
import sys
sys.path.insert(0, '/content/ATML/PA1')
from common.stages import Experiment
experiment = Experiment('/content/ATML/PA1/task1/configs/pipeline.json')
experiment.run()  # Stops after displaying the manual cue-review widget.
```

Complete visual review. If more candidates are needed, explicitly run these in separate cells:

```python
experiment.optional('expand_cues')
experiment.optional('review_cues')
```

After enough valid conflicts have been accepted, resume in a new cell in the **same kernel**:

```python
experiment.run(resume_after_review=True)
```

The original acceptance/balance assertions still run before evaluation. Expansion is optional and never runs automatically. If validation fails, complete the review before retrying the remaining stages with `experiment.run()`; in-memory state is retained. Restarting the kernel requires replaying setup; the runner does not serialize live state.

For the controlled variant, replace the manifest path with `/content/ATML/PA1/task1/controlled/configs/pipeline.json`. Keep the variants' result directories separate.

## Verification

```bash
python PA1/tools/verify_modules.py
```

This checks Python syntax and exact source parity with notebook code, ignoring only trailing whitespace at cell boundaries. It also validates source hashes and complete code-cell coverage (except dependency-install magic). It does not train models or certify numerical equivalence of a new run. The packaging was smoke-tested for ordered shared-state execution, pause/resume, and optional review stages; GPU experiments were not rerun.

# How Do Visual Models Use Cues, Adapt to Domain Shift, Generalize, and Reject Unknowns?

**Wasif Ahmed · [WasifAhmed1917](https://github.com/WasifAhmed1917)**

Advanced Topics in Machine Learning, Fall 2026 — Programming Assignment 1.

**[Read the report](report/How_Do_Visual_Models_Use_Cues__Adapt_to_Domain_Shift__Generalize__and_Reject_Unknowns_.pdf) · [CSV/JSON results and supporting artifacts on Google Drive](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing)**

## Directory structure

```text
PA1/
├── README.md
├── PYTHON_MODULES.md
├── requirements.txt
├── .gitignore
├── notebook_manifest.json
├── common/stages.py                 # shared experiment runner
├── tools/verify_modules.py          # source-parity and syntax checks
├── task1/
│   ├── ATML_PA1_Task1*.ipynb
│   ├── configs/                     # setup + pipeline.json
│   ├── data/                        # subset, transforms, cue construction/review
│   ├── models/backbones.py
│   ├── train.py
│   ├── analysis/                    # bias, similarity, representation analysis
│   ├── scripts/run_task1.py
│   ├── controlled/                  # separate Python stages for controlled variant
│   └── results/README.md
├── task2/
│   ├── ATML_PA1_Task2.ipynb
│   ├── configs/                     # setup, pipeline, Task 3 settings lock
│   ├── data/pacs.py
│   ├── models/backbone.py
│   ├── methods/alignment.py
│   ├── train.py
│   ├── evaluate_final.py
│   ├── evaluation/
│   ├── scripts/run_task2.py
│   └── results/README.md
├── task3/
│   ├── ATML_PA1_Task3.ipynb
│   ├── configs/
│   ├── data/pacs.py
│   ├── train.py
│   ├── selection/source_validation.py
│   ├── evaluate_sketch.py
│   ├── evaluation/
│   ├── scripts/run_task3.py
│   └── results/README.md
├── task4/
│   ├── ATML_PA1_Task4.ipynb
│   ├── configs/
│   ├── data/cifar10.py
│   ├── models/resnet_cifar.py
│   ├── train.py
│   ├── extract_outputs.py
│   ├── evaluate_osr.py
│   ├── evaluation/
│   ├── scripts/run_task4.py
│   └── results/README.md
└── report/
    ├── How_Do_Visual_Models_Use_Cues__Adapt_to_Domain_Shift__Generalize__and_Reject_Unknowns_.pdf
    └── figures/README.md
```

## Environment and execution

1. Open a notebook in **Google Colab** using its GitHub URL or the links below.
2. Select a **GPU runtime**. The notebooks assert CUDA availability; a CPU-only run is not supported as supplied. Use Python **3.12 or newer** for the f-string syntax in the supplied code.
3. Colab provides `google.colab`, PyTorch, and torchvision. Install missing dependencies from [requirements.txt](requirements.txt), for example after cloning the repository in Colab:

   ```python
   %pip install -r /content/ATML/PA1/requirements.txt
   ```

   The file is a dependency inventory, not an exact package lock from the original experiments. Preserve the runtime's compatible PyTorch/torchvision/CUDA installation and record installed versions with `python -m pip freeze` when reproducing a run.
4. Mount Google Drive when prompted. Outputs are written to the locations below, not automatically into this Git repository.
5. Execute cells in order, observing Task 1's manual review pauses. Complete Task 2 before Task 3; Tasks 1 and 4 are independent.

| Task | Notebook / Colab | Scope |
| --- | --- | --- |
| 1 | [AdaIN notebook](task1/ATML_PA1_Task1.ipynb) · [Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA1/task1/ATML_PA1_Task1.ipynb) | STL-10; frozen ResNet-50, ViT-B/16, CLIP; color, texture, translation, patch interventions |
| 1 variant | [Controlled notebook](task1/ATML_PA1_Task1_Controlled.ipynb) · [Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA1/task1/ATML_PA1_Task1_Controlled.ipynb) | Separate VGG19/foreground-mask cue-conflict construction |
| 2 | [Notebook](task2/ATML_PA1_Task2.ipynb) · [Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA1/task2/ATML_PA1_Task2.ipynb) | PACS source-only, DAN, DANN, CDAN |
| 3 | [Notebook](task3/ATML_PA1_Task3.ipynb) · [Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA1/task3/ATML_PA1_Task3.ipynb) | Shared ERM baseline, DAN-DG, SAM; unseen Sketch domain |
| 4 | [Notebook](task4/ATML_PA1_Task4.ipynb) · [Colab](https://colab.research.google.com/github/WasifAhmed1917/ATML/blob/main/PA1/task4/ATML_PA1_Task4.ipynb) | CIFAR-10 known classes; fixed CIFAR-100 unknown groups; Vanilla, GCSC, PROSER |

### Data preparation

- **STL-10:** downloaded by torchvision to `/content/stl10_data` in Task 1.
- **PACS:** Tasks 2–3 use the Hugging Face dataset `Azeez577/PACS`, revision `46a0a83`, archive `PACS.zip`. The notebooks verify SHA-256 `42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147` and extract to `/content/PACS_extracted`. Task 2 also supports an existing `/content/PACS` input. Task 3 expects the extracted layout.
- **CIFAR-10 / CIFAR-100:** downloaded by torchvision to `/content/data` in Task 4.
- Pretrained torchvision and OpenCLIP weights are downloaded by the libraries. The AdaIN Task 1 notebook additionally clones `naoto0804/pytorch-AdaIN` and downloads its encoder/decoder weights.

Datasets, feature caches, and model checkpoints are intentionally excluded from Git. Network access, sufficient GPU memory, and Drive storage are required.

### Reproducibility and task dependencies

The notebooks use seed **6304** and contain the split logic. Task 1 creates a stratified 80/20 training split and a balanced 500-image test subset. Tasks 2–3 share source-domain 80/20 splits and the source-only/ERM checkpoint. Task 4 uses a stratified 90/10 CIFAR-10 training/validation split.

Task 1 cue conflicts require **manual visual acceptance/rejection before evaluation**. Do not use an unattended “Run all” through the review stage. Keep the review CSV and selected IDs with the exported results. The two Task 1 notebooks are distinct experiments; keep their outputs separate and identify which variant supports each reported result.

Task 2 writes the Task 3 settings lock before its final labeled target evaluation. Task 3 reads this lock, the identical ERM checkpoint, source splits, configuration, and Task 2 summary. Its final comparison uses the Task 2 summary; do not use target results to revise the locked Task 3 settings. See each task README for exact prerequisites.

## Saved artifacts

Paths below are relative to `/content/drive/MyDrive/ATML_PA1/` except where explicitly absolute.

| Task | Runtime output location | Examples of generated artifacts |
| --- | --- | --- |
| 1 AdaIN | `/content/task1_results`; final backup cell copies to `Task1_STL10/` | `config.json`, split/test IDs, cue review, `performance.csv`, `shape_bias.csv`, translation and representation results |
| 1 controlled | `Task1_STL10_Controlled/` | Separate config, review/accepted IDs, performance tables, cue figures, t-SNE results |
| 2 | `Task2_PACS_stabilized/results/` | `config.json`, `splits/pacs_sketch_seed6304.json`, settings lock, `task2_summary.csv`, per-class metrics and failures |
| 3 | `Task3_PACS/results/` | `config.json`, `task3_summary.csv`, source diagnostics, strength study, Task 2 comparison |
| 4 | `Task4_CIFAR_OSR/` (`results/`, `arrays/`, `checkpoints/`) | `frozen_protocol.json`, `environment.json`, `vanilla_scores.csv`, `trained_models.csv`, failure tables and figures |

**Results archive:** [CSV/JSON files and supporting experiment artifacts](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing). This is the author-provided Drive folder. Its contents and anonymous-access permissions were not verified during packaging. The task `results/README.md` files point to the same archive and list expected artifacts.

**Included in Git:** five unchanged notebooks with saved cell outputs, notebook-derived Python stages, execution manifests, documentation, dependency inventory, and the supplied final report PDF. The report is uploaded unchanged. Experiment CSV/JSON exports are linked through Drive rather than copied into Git; the JSON pipeline files in this repository describe execution order and are not experiment-result exports.

The assignment also asks for machine-readable results and split/configuration artifacts in the GitHub repository. The external Drive link provides an artifact location but does not itself meet that repository-specific requirement. Small original CSV/JSON results and split indices can be added under the corresponding `results/` directories; keep Task 1 variants separate and large datasets/checkpoints outside Git.

## Attribution

- **Direct external implementation used by the AdaIN notebook:** [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN), including `net`, `adaptive_instance_normalization`, and pretrained encoder/decoder weights. The notebook records the cloned commit in `config.json`. Respect the upstream license for reused code and weights.
- **Libraries and pretrained models:** PyTorch/torchvision (models, datasets, transforms), OpenCLIP (`ViT-B-32`, `pretrained='openai'`), scikit-learn, NumPy, pandas, SciPy, Matplotlib, Pillow, OpenCV (GrabCut), tqdm, ipywidgets, and Hugging Face Hub.
- **Method references specified by the assignment:** Geirhos et al. (2019), Huang and Belongie (2017, AdaIN), Long et al. (2015, DAN; 2018, CDAN), Ganin et al. (2016, DANN), Foret et al. (2021, SAM), and Zhou et al. (2021, PROSER). Method references do not establish that source code was copied from those authors.
- This README, dependency inventory, Python stage packaging, and repository organization were prepared with coding-assistant help. The supplied notebook bytes were preserved. Any additional external code reuse in the original notebooks must be attributed by the author; the original development history was not supplied.

## Verification scope

Packaging checks validate notebook JSON, unchanged notebook/report hashes, Python syntax, exact stage-to-notebook source parity, runner pause/resume behavior, local documentation links, and Git contents. Training and evaluation have not been rerun as part of this upload; this README does not certify full experimental or report compliance.

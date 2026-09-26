# How Do Visual Models Use Cues, Adapt to Domain Shift, Generalize, and Reject Unknowns?

Wasif Ahmed — ATML, Fall 2026 — Programming Assignment 1

- [Report](report/How_Do_Visual_Models_Use_Cues__Adapt_to_Domain_Shift__Generalize__and_Reject_Unknowns_.pdf)
- [Results, CSV and JSON files](https://drive.google.com/drive/folders/1QFqu7rfbbX2YxfVg_aJJ3Z0eTg7HbkP-?usp=sharing)

## Files

- **task1:** Visual cues and representations on STL-10, including the controlled experiment.
- **task2:** Domain adaptation on PACS.
- **task3:** Domain generalization on PACS.
- **task4:** Open-set recognition on CIFAR-10 and CIFAR-100.
- **report:** Final report.

Each task folder contains its notebooks and Python files.

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

## How to run

1. Open a notebook in Google Colab and select a GPU runtime.
2. Install any missing packages listed in `requirements.txt`.
3. Mount Google Drive when prompted and run the cells in order.

Run Task 2 before Task 3, which uses its saved files. Task 1 requires manual review of the generated cue-conflict images before continuing. Tasks 1 and 4 can be run independently.

For the Python files, see [run instructions](PYTHON_MODULES.md).

Task 1 uses [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) for style transfer.

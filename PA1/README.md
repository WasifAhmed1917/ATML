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

## How to run

1. Open a notebook in Google Colab and select a GPU runtime.
2. Install any missing packages listed in `requirements.txt`.
3. Mount Google Drive when prompted and run the cells in order.

Run Task 2 before Task 3, which uses its saved files. Task 1 requires manual review of the generated cue-conflict images before continuing. Tasks 1 and 4 can be run independently.

For the Python files, see [run instructions](PYTHON_MODULES.md).

Task 1 uses [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) for style transfer.

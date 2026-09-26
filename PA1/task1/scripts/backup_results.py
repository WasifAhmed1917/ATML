# Notebook-derived stage: ATML_PA1_Task1.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 34
from google.colab import drive
from datetime import datetime, timezone
from pathlib import Path
import csv
import shutil

drive.mount('/content/drive')

source = Path('/content/task1_results')
required = [
    'config.json', 'selected_test_ids.csv', 'cue_review.csv',
    'cue_accepted_ids.csv', 'performance.csv', 'shape_bias.csv',
    'representation_stability.csv', 'per_class_clean_accuracy.csv',
]
missing = [name for name in required if not (source / name).is_file()]
assert not missing, f'Missing results: {missing}'

parent = Path('/content/drive/MyDrive/ATML_PA1/Task1_STL10')
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
destination = parent / stamp
destination.mkdir(parents=True, exist_ok=False)

def folder_for(path):
    name = path.name
    if path.parent.name == 'cue_contact_sheets' or path.suffix.lower() == '.png':
        return 'figures'
    if name.endswith('_head.pt'):
        return 'linear_heads'
    if name in {'vgg_normalised.pth', 'decoder.pth'}:
        return 'style_transfer_weights'
    if name in {'clean_subset.npz', 'cue_candidates.npz'}:
        return 'image_arrays'
    if name.startswith('cue_'):
        return 'cue_review_and_results'
    if name in {'selected_test_ids.csv', 'train_val_ids.json',
                'patch_permutations.npy', 'visualization_ids.npz'}:
        return 'image_ids_and_settings'
    if name.endswith('_tsne.csv'):
        return 'tables'
    if name in {'config.json'}:
        return 'settings'
    if path.suffix.lower() == '.csv':
        return 'tables'
    return 'other'

manifest = []
for file in sorted(source.rglob('*')):
    if not file.is_file():
        continue
    relative = file.relative_to(source)
    target = destination / folder_for(file) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, target)
    assert target.stat().st_size == file.stat().st_size
    manifest.append((str(relative), str(target.relative_to(destination)), file.stat().st_size))

with (destination / 'file_manifest.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['original_path', 'drive_path', 'bytes'])
    writer.writerows(manifest)

print(f'Saved {len(manifest)} result files to:')
print(destination)
print(f'Total: {sum(row[2] for row in manifest) / 1e9:.2f} GB')

"""Verify Python syntax, source parity, and complete notebook code-cell coverage."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify():
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    for manifest in sorted(ROOT.rglob('pipeline.json')):
        spec = json.loads(manifest.read_text())
        notebook = ROOT / spec['notebook']
        assert hashlib.sha256(notebook.read_bytes()).hexdigest() == spec['notebook_sha256'], notebook
        cells = json.loads(notebook.read_text())['cells']
        seen = []
        files = set()
        for stage in spec['stages'] + spec.get('optional_stages', []):
            path = manifest.parent.parent / stage['file']
            if path in files:  # Review widget can be explicitly displayed again.
                continue
            files.add(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == stage['sha256'], path
            expected = f"# Notebook-derived stage: {notebook.name}\n# Run through common.stages.Experiment, not as an independent module.\n"
            for index in stage['source_cells']:
                cell = cells[index]
                assert cell['cell_type'] == 'code'
                expected += f'\n# %% Original notebook cell {index}\n' + ''.join(cell['source']).rstrip() + '\n'
                seen.append(index)
            assert path.read_text() == expected, path
        expected_indices = [i for i, cell in enumerate(cells)
                            if cell['cell_type'] == 'code'
                            and not ''.join(cell['source']).startswith('%pip')]
        assert sorted(seen) == expected_indices, manifest
        print('Verified:', spec['notebook'])
    print('All Python syntax and notebook-source parity checks passed.')


if __name__ == '__main__':
    verify()

"""Run notebook-derived Python stages with one persistent experiment namespace.

Stage files deliberately retain notebook globals and side effects. They are scripts,
not independent importable APIs. Importing this runner does not start an experiment.
"""
from pathlib import Path
import json


class Experiment:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent.parent
        self.spec = json.loads(self.manifest.read_text())
        self.stages = self.spec['stages']
        self.namespace = {'__name__': '__main__'}
        self.position = 0
        self.paused = False

    def _execute(self, stage):
        path = (self.root / stage['file']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Stage path is outside the experiment directory')
        self.namespace['__file__'] = str(path)
        print(f"Running {stage['name']}: {stage['file']}")
        exec(compile(path.read_text(), str(path), 'exec'), self.namespace)

    def run(self, *, resume_after_review=False):
        """Run remaining ordered stages, stopping at a manual-review boundary."""
        if self.paused and not resume_after_review:
            raise RuntimeError('Finish visual review, then call run(resume_after_review=True).')
        self.paused = False
        while self.position < len(self.stages):
            stage = self.stages[self.position]
            self._execute(stage)
            self.position += 1
            if stage.get('pause_after'):
                self.paused = True
                print('Paused for visual review. Complete it before resuming in this session.')
                return

    def optional(self, name):
        """Run an explicitly requested optional step in the same live namespace."""
        if not self.paused:
            raise RuntimeError('Optional cue-review steps require a paused review session.')
        stage = next((s for s in self.spec.get('optional_stages', []) if s['name'] == name), None)
        if stage is None:
            raise ValueError(f'Unknown optional stage: {name}')
        self._execute(stage)


def main(manifest):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List stages without loading ML dependencies')
    args = parser.parse_args()
    experiment = Experiment(manifest)
    if args.list:
        for stage in experiment.stages:
            print(stage['name'], stage['file'], '(manual review)' if stage.get('pause_after') else '')
        return
    if any(s.get('pause_after') for s in experiment.stages):
        parser.error('Task 1 needs an interactive Colab session; use Experiment as documented in PYTHON_MODULES.md.')
    experiment.run()

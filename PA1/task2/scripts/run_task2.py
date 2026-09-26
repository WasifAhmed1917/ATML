"""Run the ordered notebook-derived experiment (use --list for inspection)."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PA1 = next(p for p in HERE.parents if (p / 'common/stages.py').is_file())
if str(PA1) not in sys.path:
    sys.path.insert(0, str(PA1))
from common.stages import main

if __name__ == '__main__':
    main(HERE.parent.parent / 'configs/pipeline.json')

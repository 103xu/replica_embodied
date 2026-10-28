"""Run from any working directory: python mujoco_teleop/run.py --help."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Use my_env Python 3.10 (or run.ps1), rather than base Python 3.14.')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(ROOT / '.downloads' / 'huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / '.downloads' / 'datasets'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.downloads' / 'matplotlib'))
sys.path.insert(0, str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor' / 'every_embodied'))

if __name__ == '__main__':
    from omy_repro.cli import main
    main()

"""Restore the isolated CPU-only pi05 data/test environment from its lockfile."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    uv = shutil.which('uv')
    if uv is None:
        raise SystemExit('uv is required for this separate environment; existing my_env is not upgraded.')
    environment = dict(os.environ, UV_CACHE_DIR=str(ROOT / '.downloads/uv-cache'),
                       UV_PYTHON_INSTALL_DIR=str(ROOT / '.downloads/python'))
    subprocess.run([uv, 'python', 'install', '3.12.13', '--no-progress'], env=environment, check=True)
    venv = ROOT / '.venv-pi05'
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([uv, 'venv', str(venv), '--python', '3.12.13'], env=environment, check=True)
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '--only-binary', ':all:',
                    '-r', str(ROOT / 'requirements-pi05-cpu.lock.txt'),
                    '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                    '--index-strategy', 'unsafe-best-match'], env=environment, check=True)
    print(f'Data conversion/tests only; this CPU environment does not run pi05 GPU training: {python}')


if __name__ == '__main__':
    main()

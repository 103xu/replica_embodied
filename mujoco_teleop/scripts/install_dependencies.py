"""Install project-local dependencies while reusing the current Python and torch."""
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_COMMIT = '10b7b3532543b4adfb65760f02a49b4c537afde7'


def main():
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Use the existing my_env Python 3.10 for this reproduction.')
    subprocess.run([sys.executable, '-c', 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'], check=True)
    dependencies = ROOT / '.deps'
    if dependencies.exists():
        raise SystemExit('Dependencies already exist. Reuse them; do not overwrite an active installation.')
    downloads = ROOT / '.downloads'
    downloads.mkdir(exist_ok=True)
    environment = dict(os.environ, PIP_CACHE_DIR=str(downloads / 'pip-cache'))
    requirements = ROOT / 'requirements-lock.txt'
    if not requirements.exists():
        requirements = ROOT / 'requirements.txt'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '--target', str(dependencies), '-r', str(requirements)], check=True, env=environment)
    archive = downloads / f'lerobot-{LEROBOT_COMMIT}.zip'
    if not archive.exists():
        with urllib.request.urlopen(f'https://codeload.github.com/huggingface/lerobot/zip/{LEROBOT_COMMIT}', timeout=120) as response:
            archive.write_bytes(response.read())
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(dependencies), str(archive)], check=True, env=environment)
    print('Project dependencies installed in .deps; existing environment packages unchanged.')


if __name__ == '__main__':
    main()

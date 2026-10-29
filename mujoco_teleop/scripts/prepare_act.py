"""Install ACT additions locally and cache the official torchvision ResNet18 weights."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Use the existing my_env Python 3.10.')
    environment = dict(os.environ, PIP_CACHE_DIR=str(ROOT / '.downloads' / 'pip-cache'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps',
                    '--target', str(ROOT / '.deps'), '-r', str(ROOT / 'requirements-act.txt')],
                   check=True, env=environment)
    weights = ROOT / '.downloads' / 'torch' / 'hub' / 'checkpoints' / 'resnet18-f37072fd.pth'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        with urllib.request.urlopen('https://download.pytorch.org/models/resnet18-f37072fd.pth', timeout=120) as response:
            payload = response.read()
        if not hashlib.sha256(payload).hexdigest().startswith('f37072fd'):
            raise RuntimeError('Official ResNet18 weight hash mismatch.')
        weights.write_bytes(payload)
    if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith('f37072fd'):
        raise RuntimeError('Cached ResNet18 weight hash mismatch.')
    print(f'ACT dependencies and verified ResNet18 weights ready: {weights}')


if __name__ == '__main__':
    main()

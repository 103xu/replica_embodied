"""Fetch the pinned, minimal OMY tutorial sources and assets; verify Git blob hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from omy_repro.assets import extract_plate


def main():
    manifest = json.loads((ROOT / 'upstream_manifest.json').read_text(encoding='utf-8'))
    destination = ROOT / 'vendor' / 'every_embodied'
    destination.mkdir(parents=True, exist_ok=True)

    def download(entry):
        path = destination / entry['path']
        if path.exists():
            content = path.read_bytes()
            if blob_hash(content) == entry['sha']:
                return
            raise RuntimeError(f'Existing file differs from pinned upstream: {path}')
        relative = manifest['path'] + entry['path']
        url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/{urllib.parse.quote(relative)}"
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        if blob_hash(content) != entry['sha']:
            raise RuntimeError(f'Hash mismatch: {relative}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, manifest['files']))
    license_url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/LICENSE"
    license_path = destination / 'LICENSE'
    if not license_path.exists():
        with urllib.request.urlopen(license_url, timeout=30) as response:
            license_path.write_bytes(response.read())

    extract_plate(destination)
    print(f"Verified {len(manifest['files'])} upstream files at {manifest['commit']}")


def blob_hash(content):
    return hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()


if __name__ == '__main__':
    main()

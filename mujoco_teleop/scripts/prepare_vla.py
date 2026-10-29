"""Install pinned VLA additions and download explicit, revision-pinned HF models."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    'lerobot/smolvla_base': '5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6',
    'HuggingFaceTB/SmolVLM2-500M-Video-Instruct': '7b375e1b73b11138ff12fe22c8f2822d8fe03467',
    'lerobot/pi0': 'e4ed526af508e58f6008b29e9e48f1098278fdb5',
    'Jeongeun/omy_pnp_smolvla': '738e2cb9235562d58842ceb8a9469ae21278bb53',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['smolvla', 'pi0'], default='smolvla')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--skip-install', action='store_true', help='Verify/download fixed files using already prepared dependencies')
    parser.add_argument('--tutorial-checkpoint', action='store_true', help='Fetch the OMY fine-tuned checkpoint linked by the tutorial, as a separate comparison')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Use my_env Python 3.10.')
    cache = ROOT / '.downloads'
    cache.mkdir(exist_ok=True)
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ.pop('HF_HUB_OFFLINE', None)
    if not args.inspect_only and not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(ROOT / '.deps'),
                        '-r', str(ROOT / 'requirements-vla.txt')], check=True,
                       env=dict(os.environ, PIP_CACHE_DIR=str(cache / 'pip-cache')))
    sys.path.insert(0, str(ROOT / '.deps'))
    from huggingface_hub import HfApi, snapshot_download
    repos = ['lerobot/smolvla_base', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'] if args.model == 'smolvla' else ['lerobot/pi0']
    if args.tutorial_checkpoint:
        if args.model != 'smolvla':
            raise ValueError('--tutorial-checkpoint applies to SmolVLA only')
        repos[0] = 'Jeongeun/omy_pnp_smolvla'
    manifest = []
    for repo in repos:
        info = HfApi().model_info(repo, revision=REVISIONS[repo], files_metadata=True)
        patterns = ['*.json', '*.model', '*.txt', '*.jinja']
        if repo != 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct':
            patterns.append('*.safetensors')
        files = [{'path': f.rfilename, 'bytes': f.size, 'lfs': f.lfs} for f in info.siblings
                 if '/' not in f.rfilename and any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in patterns)]
        entry = {'repo_id': repo, 'revision': info.sha, 'files': files}
        print(json.dumps(entry, default=str), flush=True)
        if not args.inspect_only:
            destination = cache / 'models' / repo.split('/')[-1]
            # The complete SmolVLA checkpoint already contains the VLM weights.
            snapshot_download(repo, revision=info.sha, local_dir=destination,
                              allow_patterns=[item['path'] for item in files])
            for item in files:
                local_file = destination / item['path']
                if local_file.stat().st_size != item['bytes']:
                    raise ValueError(f'Download size mismatch: {local_file.name}')
                if item['lfs']:
                    with local_file.open('rb') as handle:
                        digest = hashlib.file_digest(handle, 'sha256').hexdigest() if sys.version_info >= (3, 11) else None
                        if digest is None:
                            checksum = hashlib.sha256()
                            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                                checksum.update(block)
                            digest = checksum.hexdigest()
                    if digest != item['lfs']['sha256']:
                        raise ValueError(f'Download SHA256 mismatch: {local_file.name}')
                    item['verified_sha256'] = digest
            entry['local_dir'] = str(destination.resolve())
        manifest.append(entry)
    name = args.model + ('_tutorial' if args.tutorial_checkpoint else '')
    (cache / f'{name}_download_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()

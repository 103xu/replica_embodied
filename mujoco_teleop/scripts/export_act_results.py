"""Copy compact ACT evidence into docs; leave large checkpoints and raw rollouts local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--deployment', type=Path, nargs='*', default=[])
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs' / 'act')
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'training_run': str(args.run.resolve()), 'files': {}}
    for name in ('training_run.json', 'metrics.jsonl', 'training_loss.png', 'offline_actions.png'):
        source = args.run / name
        if source.exists():
            shutil.copyfile(source, destination / name)
    for label in ('best', 'final'):
        for name in ('config.json', 'model.safetensors', 'evaluation.json'):
            source = args.run / label / name
            if source.exists():
                manifest['files'][f'{label}/{name}'] = {'bytes': source.stat().st_size, 'sha256': file_hash(source)}
                if name != 'model.safetensors':
                    shutil.copyfile(source, destination / f'{label}_{name}')
    for rollout in args.deployment:
        label = rollout.name
        for name in ('deployment_report.json', 'trial_000_agent.png', 'trial_000_wrist.png'):
            source = rollout / name
            if source.exists():
                shutil.copyfile(source, destination / f'{label}_{name}')
    (destination / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()

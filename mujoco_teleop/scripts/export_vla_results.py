"""Export small, completed VLA evidence; never copy model weights or datasets into Git."""
import argparse
import json
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='smolvla_omy_2000')
    parser.add_argument('--deployment', default='smolvla_2000_seed0')
    args = parser.parse_args()
    runpy.run_path(str(ROOT / 'run.py'), run_name='bootstrap')
    destination = ROOT / 'docs/vla_results'
    destination.mkdir(exist_ok=True)
    pairs = {
        ROOT / 'data/language_demo_020/reproduction.json': 'language_collection.json',
        ROOT / 'outputs/language020_audit.json': 'language_audit.json',
        ROOT / 'outputs/language020_red/replay_report.json': 'language_red_replay.json',
        ROOT / 'outputs/language020_blue_targetfixed/replay_report.json': 'language_blue_replay.json',
        ROOT / 'outputs/vla_preflight.json': 'vla_preflight.json',
        ROOT / 'outputs/smolvla_2000_diagnosis.json': 'smolvla_2000_diagnosis.json',
        ROOT / 'outputs/smolvla_chunk50_diagnosis.json': 'smolvla_chunk50_diagnosis.json',
        ROOT / 'outputs/pi05_source_replay.json': 'pi05_source_replay.json',
        ROOT / 'data/pi05_v3_001/omy_manifest.json': 'pi05_migration.json',
        ROOT / '.downloads/pi05_tests.log': 'pi05_tests.log',
        ROOT / '.downloads/tests_vla.log': 'local_tests.log',
        ROOT / '.downloads/smolvla_download_manifest.json': 'pretrained_download_manifest.json',
        ROOT / '.downloads/smolvla_tutorial_download_manifest.json': 'tutorial_model_download_manifest.json',
        ROOT / '.downloads/pi05_conversion.log': 'pi05_conversion.log',
        ROOT / 'outputs/act_vla_regression/deployment_report.json': 'act_runtime_regression.json',
    }
    for source, name in pairs.items():
        if source.is_file():
            shutil.copyfile(source, destination / name)
    run = ROOT / 'checkpoints' / args.run
    report = json.loads((run / 'training_run.json').read_text(encoding='utf-8'))
    if not report.get('complete'):
        raise RuntimeError(f'Training is not complete: {run}')
    for source, name in ((run / 'training_run.json', f'{args.run}_training.json'),
                         (run / 'metrics.jsonl', f'{args.run}_metrics.jsonl')):
        shutil.copyfile(source, destination / name)
    deployment = ROOT / 'outputs' / args.deployment
    if (deployment / 'deployment_report.json').is_file():
        shutil.copyfile(deployment / 'deployment_report.json', destination / f'{args.deployment}_deployment.json')
        for color in ('red', 'blue'):
            for extension in ('.gif', '_agent.png'):
                source = deployment / f'{color}{extension}'
                if source.exists():
                    shutil.copyfile(source, destination / f'{args.deployment}_{color}{extension}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot([m['step'] for m in metrics], [m['loss'] for m in metrics], label='sampled training flow loss')
    if all('heldout_flow_loss' in m for m in metrics):
        axis.plot([m['step'] for m in metrics], [m['heldout_flow_loss'] for m in metrics], label='fixed-noise heldout flow loss')
    axis.set(xlabel='optimizer update', ylabel='flow matching loss', title=args.run)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination / f'{args.run}_loss.png', dpi=130)
    plt.close(fig)
    print(f'Exported completed reports to {destination}')


if __name__ == '__main__':
    main()

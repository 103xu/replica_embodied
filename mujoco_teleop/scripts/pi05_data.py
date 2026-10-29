"""Run pi05 OMY data conversion in the separate Python 3.12 environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/every_embodied'


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use mujoco_teleop/.venv-pi05/Scripts/python.exe (Python 3.12).')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('test')
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--destination', type=Path, required=True)
    prepare.add_argument('--labels-verified', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HOME'] = str(ROOT / '.downloads/hf-pi05')
    # datasets embeds its cache path into lock filenames. Keep it short on
    # Windows rather than requiring global long-path registry changes.
    os.environ['HF_DATASETS_CACHE'] = str(ROOT.parent / '.hf3')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    temporary = ROOT / '.downloads/tmp-pi05'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(temporary)
    if args.command == 'test':
        return subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_pi05_omy'], cwd=VENDOR).returncode
    sys.path.insert(0, str(VENDOR))
    from pi05_omy_utils import prepare_dataset, verify_alignment, training_command
    source = args.source.resolve()
    provenance = json.loads((source / 'reproduction.json').read_text(encoding='utf-8'))
    if provenance.get('observation_timing') != 'before_command' or provenance.get('action_type') != 'absolute_joint_targets':
        raise ValueError('This source has not recorded the corrected action contract.')
    if provenance.get('state_type') == 'joint_angle':
        raise ValueError('This pi05 route requires single-cup EEF state, not the dual-cup joint state.')
    result = prepare_dataset(source, args.destination, 'local/omy-pi05', labels_verified=args.labels_verified)
    result['alignment'] = verify_alignment(args.destination)
    result['proposed_training_command'] = training_command(args.destination, ROOT / 'checkpoints/pi05_omy')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

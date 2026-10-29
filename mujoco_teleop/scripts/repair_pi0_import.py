"""Repair one known type-only import in the pinned, project-local LeRobot copy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'lerobot/common/policies/pi0/paligemma_with_expert.py'
OLD = b'from pytest import Cache'
NEW = b'from transformers.cache_utils import Cache'


def repair(dependencies):
    path = Path(dependencies) / RELATIVE
    original = path.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    if OLD in original:
        if original.count(OLD) != 1:
            raise ValueError('Unexpected pinned source: refusing an ambiguous patch')
        path.write_bytes(original.replace(OLD, NEW, 1))
        changed = True
    elif NEW in original:
        changed = False
    else:
        raise ValueError('Pinned Cache import changed; inspect source before patching')
    return {'path': RELATIVE, 'changed': changed, 'sha256_before': before,
            'sha256_after': hashlib.sha256(path.read_bytes()).hexdigest(),
            'replacement': 'pytest.Cache -> transformers.cache_utils.Cache (type annotation only)'}


def main():
    report = repair(ROOT / '.deps')
    destination = ROOT / '.downloads/pi0_import_patch.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

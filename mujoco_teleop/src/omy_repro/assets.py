"""Unpack the large plate mesh locally; keep its smaller upstream ZIP in Git."""
from pathlib import Path
import zipfile


def extract_plate(vendor):
    root = (Path(vendor) / 'asset' / 'objaverse').resolve()
    archive_path = root / 'plate_11.zip'
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if '__MACOSX' in relative.parts or relative.name == '.DS_Store':
                continue
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe archive member: {entry.filename}')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))


def ensure_assets(vendor):
    # Checking every archive member also repairs interrupted partial extraction.
    extract_plate(vendor)

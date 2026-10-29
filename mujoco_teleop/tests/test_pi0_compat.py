"""The pinned PI0 type import repair is small, bounded and repeatable."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_pi0_import', PROJECT / 'scripts/repair_pi0_import.py')
repair_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair_module)


class PI0CompatibilityTests(unittest.TestCase):
    def test_repairs_only_the_known_import_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as directory:
            path = Path(directory) / repair_module.RELATIVE
            path.parent.mkdir(parents=True)
            source = b'# preserved\r\nfrom pytest import Cache\r\nmarker = 42\r\n'
            path.write_bytes(source)
            self.assertTrue(repair_module.repair(directory)['changed'])
            self.assertEqual(path.read_bytes(), source.replace(repair_module.OLD, repair_module.NEW))
            self.assertFalse(repair_module.repair(directory)['changed'])
            path.write_bytes(b'unrecognized source')
            with self.assertRaises(ValueError):
                repair_module.repair(directory)
            self.assertEqual(path.read_bytes(), b'unrecognized source')

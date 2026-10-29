"""A missing pretrained core must never be accepted as successful fine-tuning."""
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT / 'run.py'), run_name='bootstrap')

import torch
from safetensors.torch import save_file
from omy_repro.smolvla import transfer_core


class PretrainedTransferTests(unittest.TestCase):
    def test_all_core_weights_required_and_foreign_normalizers_ignored(self):
        policy = SimpleNamespace(model=torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            checkpoint = Path(folder) / 'model.safetensors'
            save_file({'model.weight': torch.ones(2, 2),
                       'normalize_inputs.foreign_robot_stat': torch.ones(99)}, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'Pretrained core mismatch'):
                transfer_core(policy, checkpoint)
            save_file({'model.weight': torch.ones(2, 2), 'model.bias': torch.ones(2),
                       'normalize_inputs.foreign_robot_stat': torch.ones(99)}, str(checkpoint))
            report = transfer_core(policy, checkpoint)
            self.assertEqual(report['core_tensors_loaded'], 2)
            torch.testing.assert_close(policy.model.weight, torch.ones(2, 2))
            save_file({'model.weight': torch.ones(2, 2), 'model.bias': torch.ones(2),
                       'model.unexpected_projection': torch.ones(1)}, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'Pretrained core mismatch'):
                transfer_core(policy, checkpoint)


if __name__ == '__main__':
    unittest.main(verbosity=2)

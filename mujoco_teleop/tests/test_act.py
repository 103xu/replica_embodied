"""ACT action chunking, leakage-free evaluation and strict checkpoint round trips."""
from pathlib import Path
import runpy
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT / 'run.py'), run_name='bootstrap')

import torch
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from omy_repro.act import chunk_indices, initialize_from_checkpoint, load_policy, predict_chunks, seed_everything


class ACTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        config = ACTConfig(device='cpu', chunk_size=3, n_action_steps=3,
                           dim_model=32, dim_feedforward=64, n_heads=4,
                           n_encoder_layers=1, n_decoder_layers=1,
                           n_vae_encoder_layers=1, latent_dim=8,
                           pretrained_backbone_weights=None,
                           input_features={'observation.image': PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
                                           'observation.state': PolicyFeature(FeatureType.STATE, (6,))},
                           output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))})
        stats = {key: {'mean': torch.zeros(shape), 'std': torch.ones(shape)} for key, shape in
                 [('observation.image', (3, 1, 1)), ('observation.state', (6,)), ('action', (7,))]}
        cls.policy = ACTPolicy(config, dataset_stats=stats).eval()
        cls.stats = stats
        cls.inputs = {'observation.image': torch.rand(2, 3, 32, 32),
                      'observation.state': torch.rand(2, 6)}

    def test_chunks_do_not_cross_episode_boundaries(self):
        indices, padding = chunk_indices([3, 3, 3, 5, 5], 3)
        self.assertEqual(indices.tolist(), [[0, 1, 2], [1, 2, 2], [2, 2, 2], [3, 4, 4], [4, 4, 4]])
        self.assertEqual(padding.tolist(), [[False, False, False], [False, False, True],
                                           [False, True, True], [False, False, True], [False, True, True]])

    def test_inference_ignores_ground_truth_future_actions(self):
        first = predict_chunks(self.policy, {**self.inputs, 'action': torch.zeros(2, 3, 7)})
        second = predict_chunks(self.policy, {**self.inputs, 'action': torch.ones(2, 3, 7) * 100})
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_trained_checkpoint_roundtrip_keeps_predictions(self):
        expected = predict_chunks(self.policy, self.inputs)
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            self.policy.save_pretrained(folder)
            restored = load_policy(folder, device='cpu')
            actual = predict_chunks(restored, self.inputs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_missing_weights_cannot_fall_back_to_random_initialization(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            with self.assertRaises(FileNotFoundError):
                load_policy(folder, device='cpu')

    def test_longer_chunks_preserve_trained_nonpositional_weights(self):
        import copy
        config = copy.deepcopy(self.policy.config)
        config.chunk_size = config.n_action_steps = 5
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            self.policy.save_pretrained(folder)
            transferred, details = initialize_from_checkpoint(config, self.stats, folder, 'cpu')
            torch.testing.assert_close(transferred.model.action_head.weight,
                                       self.policy.model.action_head.weight, atol=0, rtol=0)
            self.assertEqual(transferred.model.decoder_pos_embed.weight.shape[0], 5)
            self.assertIn('model.decoder_pos_embed.weight', details['resized_keys'])
            self.assertEqual(predict_chunks(transferred.eval(), self.inputs).shape, (2, 5, 7))


if __name__ == '__main__':
    unittest.main(verbosity=2)

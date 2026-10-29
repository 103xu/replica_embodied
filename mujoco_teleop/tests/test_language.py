"""Guard language labels, joint-state semantics and repeatable paired scenes."""
from pathlib import Path
import runpy
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT / 'run.py'), run_name='bootstrap')

import numpy as np
from omy_repro.language import load_language, task_status, physical_candidate
from omy_repro.runtime import LanguageEnvironment, UpstreamLanguageEnvironment
from omy_repro.smolvla import Frames


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = LanguageEnvironment(seed=0, headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_instruction_changes_target_without_changing_scene(self):
        env = self.env
        env.reset(seed=4)
        env.set_instruction('Place the red mug on the plate.')
        red_snapshot = env.snapshot()
        red_image = env.images()[0]
        env.set_instruction('Place the blue mug on the plate.')
        self.assertEqual(env.obj_target, 'body_obj_mug_6')
        np.testing.assert_array_equal(red_snapshot, env.snapshot())
        np.testing.assert_array_equal(red_image, env.images()[0])
        env.reset(seed=4)
        np.testing.assert_array_equal(red_snapshot, env.snapshot())

    def test_joint_state_and_recorded_command_are_distinct(self):
        env = self.env
        env.reset(seed=0)
        measured = env.get_state()
        np.testing.assert_array_equal(measured, env.env.get_qpos_joints(env.joint_names).astype(np.float32))
        snapshot = env.snapshot()
        env.step(np.array([0.003, 0, -0.003, 0, 0, 0, 0.8], dtype=np.float32))
        np.testing.assert_array_equal(snapshot, env.snapshot())
        command = env.get_commanded_joint_action()
        self.assertEqual(command.shape, (7,))
        self.assertAlmostEqual(float(command[-1]), 0.03, places=6)
        self.assertGreater(float(np.linalg.norm(command[:6] - measured)), 0)

    def test_single_cup_data_cannot_be_silently_used_as_joint_state(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            with self.assertRaises(ValueError):
                load_language(folder)

    def test_physical_rule_is_independent_of_old_ten_centimeter_confirmation(self):
        status = {'legacy_success': True, 'upright': True, 'strict_success': False,
                  'mug_lift_height_m': 0.058}
        self.assertTrue(physical_candidate(status, 3))
        self.assertFalse(physical_candidate(status, 2))
        self.assertFalse(physical_candidate({**status, 'upright': False}, 3))
        self.assertFalse(physical_candidate({**status, 'legacy_success': False}, 3))

    def test_legacy_success_follows_language_target(self):
        from unittest.mock import patch
        env = self.env
        positions = {'body_obj_plate_11': np.array([0.3, -0.25, 0.82]),
                     'body_obj_mug_5': np.array([0.3, -0.25, 0.86]),
                     'body_obj_mug_6': np.array([0.3, 0.2, 0.86]),
                     'tcp_link': np.array([0.3, 0, 1.0])}
        with patch.object(env.env, 'get_p_body', side_effect=positions.__getitem__), patch.object(env.env, 'get_qpos_joint', return_value=np.array([0.0])):
            env.set_instruction('Place the red mug on the plate.')
            self.assertTrue(env.check_success())
            env.set_instruction('Place the blue mug on the plate.')
            self.assertFalse(env.check_success())

    def test_published_model_interface_preserves_original_gripper_command(self):
        env = UpstreamLanguageEnvironment(seed=0, headless=True, action_type='joint_angle')
        try:
            command = env.get_commanded_joint_action()
            command[-1] = 0.8
            state = env.snapshot()
            env.step(command)
            np.testing.assert_allclose(env.q[-4:], [0.8, 0.64, 0.8, 0.64], atol=1e-6)
            np.testing.assert_array_equal(state, env.snapshot())
        finally:
            env.close()

    def test_split_keeps_whole_episodes_and_stats_ignore_holdout(self):
        import torch
        from types import SimpleNamespace
        class Table:
            def select_columns(self, keys):
                return self
            def with_format(self, fmt):
                return self
            def __getitem__(self, key):
                return {'observation.state': [[0.0] * 6, [2.0] * 6, [100.0] * 6],
                        'action': [[0.0] * 7, [2.0] * 7, [100.0] * 7]}
        dataset = SimpleNamespace(hf_dataset=Table(), num_frames=3, num_episodes=2,
                                  episode_data_index={'from': [0, 2], 'to': [2, 3]})
        frames = Frames(dataset, 3, 1)
        self.assertEqual(frames.train_indices.tolist(), [0, 1])
        self.assertEqual(frames.eval_indices.tolist(), [2])
        self.assertEqual(frames.indices[1].tolist(), [1, 1, 1])
        torch.testing.assert_close(frames.stats()['observation.state']['mean'], torch.ones(6))


if __name__ == '__main__':
    unittest.main(verbosity=2)

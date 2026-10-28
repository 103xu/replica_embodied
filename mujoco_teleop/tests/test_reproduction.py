"""Checks for the action contract, dataset round trip and safe output handling."""
from pathlib import Path
import runpy
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT / 'run.py'), run_name='bootstrap')

import numpy as np
from omy_repro.dataset import add_frame, create_dataset, load_dataset
from omy_repro.runtime import OMYEnvironment


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = OMYEnvironment(seed=0, headless=True, action_type='joint_angle')

    @classmethod
    def tearDownClass(cls):
        cls.environment.close()

    def setUp(self):
        self.environment.reset(seed=0)

    def test_existing_data_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            marker = Path(folder) / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                create_dataset(folder)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_label_is_command_and_does_not_alias_next_command(self):
        env = self.environment
        measured = env.env.get_qpos_joints(joint_names=env.joint_names).copy()
        target = measured + np.array([0.02, 0, 0, 0, 0, 0], dtype=np.float32)
        env.step(np.r_[target, 1].astype(np.float32))
        label = env.get_commanded_joint_action()
        np.testing.assert_allclose(label[:6], target, atol=1e-6)
        self.assertGreater(float(np.linalg.norm(label[:6] - measured)), 0.01)
        self.assertAlmostEqual(float(label[-1]), 0.03, places=6)
        np.testing.assert_array_equal(env.q[:6], label[:6])
        self.assertEqual(env.q.dtype, np.dtype('float32'))
        env.step(np.r_[measured, 0].astype(np.float32))
        np.testing.assert_allclose(label[:6], target, atol=1e-6)

    def test_physics_snapshot_replays_identical_joint_commands(self):
        env = self.environment
        state = env.snapshot()
        target = env.get_commanded_joint_action().copy()
        target[0] += 0.01
        env.step(target)
        env.advance()
        expected = env.env.data.qpos.copy()
        env.restore(state)
        env.step(target)
        env.advance()
        np.testing.assert_allclose(env.env.data.qpos, expected, atol=1e-8)

    def test_ik_command_preparation_preserves_full_integration_state(self):
        env = self.environment
        before = env.snapshot()
        env.action_type = 'eef_pose'
        try:
            env.step(np.array([0.003, 0.002, -0.003, 0, 0, 0, 0], dtype=np.float32))
            np.testing.assert_array_equal(env.snapshot(), before)
        finally:
            env.action_type = 'joint_angle'

    def test_lerobot_parquet_roundtrip_keeps_images_and_action_labels(self):
        env = self.environment
        with tempfile.TemporaryDirectory(dir=PROJECT / '.downloads') as folder:
            root = Path(folder) / 'dataset'
            dataset = create_dataset(root)
            actions, images = [], []
            try:
                for i in range(3):
                    state = env.get_ee_pose().copy()
                    agent, wrist = env.images()
                    action = env.get_commanded_joint_action().copy()
                    action[0] += 0.005
                    env.step(action)
                    add_frame(dataset, env, agent, wrist, state)
                    actions.append(env.get_commanded_joint_action().copy())
                    images.append(agent)
                    env.advance()
                dataset.save_episode()
            finally:
                dataset.stop_image_writer()
            reloaded = load_dataset(root)
            self.assertEqual(reloaded.num_frames, 3)
            self.assertEqual(reloaded.num_episodes, 1)
            for i in range(3):
                frame = reloaded[i]
                np.testing.assert_allclose(frame['action'].numpy(), actions[i])
                image = (frame['observation.image'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                np.testing.assert_array_equal(image, images[i])
                self.assertAlmostEqual(float(frame['timestamp']), i / 20, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Use the tutorial's pinned LeRobot v2.1 dataset implementation."""
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    'observation.image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channels']},
    'observation.wrist_image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channels']},
    'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': ['state']},
    'action': {'dtype': 'float32', 'shape': (7,), 'names': ['action']},
    'obj_init': {'dtype': 'float32', 'shape': (6,), 'names': ['obj_init']},
}
TASK = 'Put mug cup on the plate'
REPO_ID = 'datawhale_eai_pnp'


def create_dataset(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}. Choose a new --dataset directory.')
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id=REPO_ID, root=root, robot_type='omy', fps=20,
                                features=FEATURES, use_videos=False,
                                image_writer_processes=0, image_writer_threads=2)


def load_dataset(root):
    root = Path(root).resolve()
    if not (root / 'meta' / 'info.json').is_file():
        raise FileNotFoundError(f'No local dataset at {root}. Collect an episode first.')
    # Prevent accidental downloads if a local episode or metadata is missing.
    return LeRobotDataset(repo_id=REPO_ID, root=root, video_backend='pyav')


def add_frame(dataset, env, agent, wrist, state):
    dataset.add_frame({'observation.image': agent, 'observation.wrist_image': wrist,
                       'observation.state': np.asarray(state, dtype=np.float32),
                       'action': env.get_commanded_joint_action().copy(),
                       'obj_init': env.obj_init_pose.copy()}, task=TASK)


def write_run(root, details):
    (Path(root) / 'reproduction.json').write_text(json.dumps(details, indent=2), encoding='utf-8')

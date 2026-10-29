"""A deterministic IK controller to validate collection/replay without manual input.

This is not a learned VLA policy. Objects move only through MuJoCo physics.
"""
import numpy as np


def scripted_actions(env):
    poses = [x.copy() for x in env.get_obj_pose()]
    if len(poses) == 3:
        mug = poses[0 if env.obj_target == 'body_obj_mug_5' else 1]
        plate = poses[2]
    else:
        mug, plate = poses

    def move(target, grip=0.0, count=None):
        target = np.asarray(target, dtype=np.float32)
        for _ in range(count or 180):
            delta = target - env.p0
            if count is None and np.linalg.norm(delta) < 0.002:
                break
            delta = np.clip(delta, -0.003, 0.003)
            yield np.array([*delta, 0, 0, 0, grip], dtype=np.float32)
        for _ in range(15):
            yield np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float32)

    phases = [
        ('approach', lambda: move([mug[0], mug[1], 1.05])),
        # OMY TCP is below the finger pads. Closing at the mug center is too high.
        ('descend', lambda: move([mug[0], mug[1], mug[2] - 0.03])),
        ('close', lambda: move(env.p0.copy(), 0.8, count=40)),
        ('lift', lambda: move([mug[0], mug[1], 1.08], 0.8)),
        ('transfer', lambda: move([plate[0], plate[1], 1.08], 0.8)),
        ('lower', lambda: move([plate[0], plate[1], 0.91], 0.8)),
        ('release', lambda: move(env.p0.copy(), 0.0, count=40)),
        ('retreat', lambda: move([plate[0], plate[1], 1.08])),
    ]
    for phase, actions in phases:
        print(f'Phase: {phase}', flush=True)
        for action in actions():
            yield phase, action

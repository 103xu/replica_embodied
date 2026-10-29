"""Adapt the unmodified upstream OMY environment for GUI and offscreen use."""
from pathlib import Path
import time

import mujoco
import glfw
import numpy as np
from mujoco_env.y_env import SimpleEnv
from .assets import ensure_assets

PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / 'vendor' / 'every_embodied'
SCENE = VENDOR / 'asset' / 'example_scene_y.xml'
FPS = 20


class OMYEnvironment(SimpleEnv):
    def __init__(self, seed=0, headless=False, action_type='eef_pose'):
        ensure_assets(VENDOR)
        self.headless = headless
        self.camera_renderer = None
        super().__init__(str(SCENE), seed=seed, action_type=action_type, state_type='ee_pose')
        # A direct renderer works for GUI and offscreen tests alike.
        self.camera_renderer = mujoco.Renderer(self.env.model, height=256, width=256)
        self.substeps = round(1 / FPS / self.env.model.opt.timestep)
        if not np.isclose(self.substeps * self.env.model.opt.timestep, 1 / FPS):
            raise ValueError('Simulation timestep must divide the 20 Hz control period.')

    def init_viewer(self):
        if self.headless:
            self.env.reset()
        else:
            super().init_viewer()

    def images(self):
        frames = []
        for camera in (self.agent_cam_name, self.wrist_cam_name):
            self.camera_renderer.update_scene(self.env.data, camera=camera)
            frames.append(self.camera_renderer.render().copy())
        return frames

    def advance(self):
        for _ in range(self.substeps):
            self.step_env()

    def step(self, action):
        # Upstream IK restores qpos/qvel/act, but mj_forward also changes the
        # solver warm-start accelerations. Keep command preparation physically
        # side-effect free so saved joint commands can reproduce the trajectory.
        integration = self.snapshot()
        observation = super().step(action)
        # LeRobot stores float32 targets. Execute those exact targets too, rather
        # than an unrecorded float64 IK solution (contact dynamics amplify the
        # rounding difference). Rebuild the joint command without a second slew.
        command = self.get_commanded_joint_action()
        self.gripper_cmd_scalar = float(command[-1])
        action_type = self.action_type
        try:
            self.action_type = 'joint_angle'
            super().step(command)
        finally:
            self.action_type = action_type
        self.restore(integration)
        return observation

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.env.model, spec), dtype=np.float64)
        mujoco.mj_getState(self.env.model, self.env.data, state, spec)
        return state

    def restore(self, state):
        mujoco.mj_setState(self.env.model, self.env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.env.model, self.env.data)
        # mj_forward refreshes derived transforms but can change warm-start data.
        mujoco.mj_setState(self.env.model, self.env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)

    def alive(self):
        return self.headless or self.env.is_viewer_alive()

    def show(self, agent, wrist):
        if not self.headless:
            # Camera Renderer owns a different OpenGL context. The upstream
            # viewer switches only during construction; rendering in the wrong
            # context corrupts subsequent policy camera images.
            glfw.make_context_current(self.env.viewer.window)
            self.rgb_agent, self.rgb_ego = agent, wrist
            self.rgb_side = agent
            self.render(teleop=True)

    def close(self):
        if self.camera_renderer is not None:
            self.camera_renderer.close()
            self.camera_renderer = None
        if not self.headless:
            glfw.make_context_current(self.env.viewer.window)
            self.env.close_viewer()


def pace(start, fast):
    if not fast:
        time.sleep(max(0, 1 / FPS - (time.perf_counter() - start)))

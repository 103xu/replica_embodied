"""Adapt the unmodified upstream OMY environment for GUI and offscreen use."""
from pathlib import Path
import time

import mujoco
import glfw
import numpy as np
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from mujoco_env.y_env2 import SimpleEnv2
from .assets import ensure_assets

PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / 'vendor' / 'every_embodied'
SCENE = VENDOR / 'asset' / 'example_scene_y.xml'
FPS = 20


class RuntimeAdapter:
    scene = SCENE
    state_type_name = 'ee_pose'
    camera_size = (256, 256)

    def __init__(self, seed=0, headless=False, action_type='eef_pose'):
        ensure_assets(VENDOR)
        self.headless = headless
        self.camera_renderer = None
        super().__init__(str(self.scene), seed=seed, action_type=action_type, state_type=self.state_type_name)
        # A direct renderer works for GUI and offscreen tests alike.
        width, height = self.camera_size
        self.env.model.vis.global_.offwidth = max(self.env.model.vis.global_.offwidth, width)
        self.env.model.vis.global_.offheight = max(self.env.model.vis.global_.offheight, height)
        self.camera_renderer = mujoco.Renderer(self.env.model, height=height, width=width)
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
        for camera in ('agentview', 'egocentric'):
            self.camera_renderer.update_scene(self.env.data, camera=camera)
            frame = self.camera_renderer.render().copy()
            if self.camera_size != (256, 256):
                frame = np.asarray(Image.fromarray(frame).resize((256, 256))).copy()
            frames.append(frame)
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


class OMYEnvironment(RuntimeAdapter, SimpleEnv):
    """Sections 1-4: single cup, six-dimensional end-effector state."""


class LanguageEnvironment(RuntimeAdapter, SimpleEnv):
    """Sections 5-8: two cups, six-dimensional joint state."""
    scene = VENDOR / 'asset' / 'example_scene_y2.xml'
    state_type_name = 'joint_angle'
    # Use the upstream language scene/reset and labels, with the corrected OMY
    # joint/gripper command mapping already used by sections 1-4.
    def reset(self, seed=None):
        # Remove previous-episode velocities, controls and simulation time.
        self.env.reset()
        SimpleEnv2.reset(self, seed=seed)

    set_instruction = SimpleEnv2.set_instruction
    get_obj_pose = SimpleEnv2.get_obj_pose
    set_obj_pose = SimpleEnv2.set_obj_pose
    render = SimpleEnv2.render
    check_success = SimpleEnv2.check_success

    def get_state(self):
        return self.get_joint_state()[:6].copy()


class UpstreamLanguageEnvironment(RuntimeAdapter, SimpleEnv2):
    """Original section-5 gripper mapping/integrator for published-model comparison."""
    scene = VENDOR / 'asset' / 'example_scene_y2.xml'
    state_type_name = 'joint_angle'
    camera_size = (800, 600)

    def reset(self, seed=None):
        self.env.reset()
        SimpleEnv2.reset(self, seed=seed)

    def get_state(self):
        return self.get_joint_state()[:6].copy()


def pace(start, fast):
    if not fast:
        time.sleep(max(0, 1 / FPS - (time.perf_counter() - start)))

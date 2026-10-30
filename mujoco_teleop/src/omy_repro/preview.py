"""A presentation camera independent of the policy observation cameras."""
import mujoco
from PIL import Image, ImageDraw, ImageFont


class OverviewPreview:
    def __init__(self, env):
        self.env = env
        model = env.env.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
        self.renderer = mujoco.Renderer(model, width=640, height=480)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.25, 0.0, 1.0)
        self.camera.distance = 1.8
        self.camera.azimuth = 135
        self.camera.elevation = -25
        try:
            self.font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
        except OSError:
            self.font = ImageFont.load_default()

    def frame(self):
        self.renderer.update_scene(self.env.env.data, camera=self.camera)
        image = Image.fromarray(self.renderer.render().copy())
        canvas = Image.new('RGB', (640, 516), 'white')
        canvas.paste(image, (0, 36))
        ImageDraw.Draw(canvas).text((14, 7), self.env.instruction, fill='black', font=self.font)
        return canvas

    def close(self):
        self.renderer.close()

import numpy as np
import cv2
import colorsys
from vibert.lib.utils.renderer import Renderer


class HoloSmplRenderer():
    def __init__(self, params):
        self.mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        self.width = params['width']
        self.height = params['height']

        self.renderer = Renderer(resolution=(self.width, self.height), orig_img=True,
                                 wireframe=params['wireframe'], gender=params['gender'],
                                 smpl_model_dir=params['smpl_model_dir'],
                                 joint_regressor_path=params['joint_regressor_path'])

    def render(self, verts, cam, resize_to=None):
        if not isinstance(verts, np.ndarray):
            scene_verts = verts.numpy()[0]
        else:
            scene_verts = verts[0]

        # Render:
        scenec_cam = cam[0]
        img = np.zeros((self.height,self.width,3), dtype=np.uint8)
        rendered_smpl = self.renderer.render(img, scene_verts, cam=scenec_cam, color=self.mesh_color)

        if resize_to is not None:
            rendered_smpl = cv2.resize(rendered_smpl, resize_to)

        return rendered_smpl

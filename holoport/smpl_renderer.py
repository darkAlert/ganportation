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

    def render(self, data, resize_to_frame=True):
        if not 'verts' in data['vibe_output']:
            # predict vertices if they are missed:
            raise NotImplementedError
        else:
            if not isinstance(data['vibe_output']['verts'], np.ndarray):
                verts = data['vibe_output']['verts'].numpy()[0]
            else:
                verts = data['vibe_output']['verts'][0]

        # Render:
        cam = data['scene_cam'][0]
        img = np.zeros((self.height,self.width,3), dtype=np.uint8)
        rendered_smpl = self.renderer.render(img, verts, cam=cam, color=self.mesh_color)

        if resize_to_frame:
            h,w = data['frame'].shape[:2]
            rendered_smpl = cv2.resize(rendered_smpl, (w, h))

        return rendered_smpl

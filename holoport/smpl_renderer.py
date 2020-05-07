import numpy as np
import cv2
import colorsys
import torch
from vibert.lib.utils.renderer import Renderer
from lwganrt.utils.nmr import SMPLRenderer


class TrimeshSmplRenderer():
    def __init__(self, args):
        self.mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        self.width = args['width']
        self.height = args['height']

        self.renderer = Renderer(resolution=(self.width, self.height), orig_img=True,
                                 wireframe=args['wireframe'], gender=args['gender'],
                                 smpl_model_dir=args['smpl_model_dir'],
                                 joint_regressor_path=args['joint_regressor_path'])

    def render(self, verts, cam, resize_to=None):
        if not isinstance(verts, np.ndarray):
            scene_verts = verts.numpy()[0]
        else:
            scene_verts = verts[0]

        # Render:
        scene_cam = cam[0]
        img = np.zeros((self.height,self.width,3), dtype=np.uint8)
        rendered_smpl = self.renderer.render(img, scene_verts, cam=scene_cam, color=self.mesh_color)

        if resize_to is not None:
            rendered_smpl = cv2.resize(rendered_smpl, resize_to)

        return rendered_smpl


class NeuralSmplRenderer():
    def __init__(self, args):
        self.device = torch.device('cuda:' + str(args['gpu_id']))
        self.renderer = SMPLRenderer(face_path=args['smpl_faces'],
                                     uv_map_path=args['uv_mapping'],
                                     image_size=args['image_size'],
                                     tex_size=args['tex_size'],
                                     has_front=args['front_warp'],
                                     fill_back=False,
                                     part_info=args['part_info'],
                                     front_info=args['front_info'],
                                     head_info=args['head_info'],
                                     device=self.device)
        self.texs = self.renderer.debug_textures().to(self.device)[None]

    def render(self, verts, cam, resize_to=None):
        if isinstance(verts, np.ndarray):
            scene_verts = torch.from_numpy(verts)
        elif torch.is_tensor(verts):
            scene_verts = verts
        else:
            raise NotImplementedError

        if isinstance(cam, np.ndarray):
            scene_cam = torch.from_numpy(cam)
        elif torch.is_tensor(cam):
            scene_cam = cam
        else:
            raise NotImplementedError

        if scene_cam.shape[1] == 4:
            # sx, sy, tx, ty -> sx, tx, ty
            scene_cam = torch.FloatTensor([[scene_cam[0, 0], scene_cam[0, 2], scene_cam[0, 3]]])

        scene_verts = scene_verts.to(self.device).float()
        scene_cam = scene_cam.to(self.device).float()

        # Render:
        img, _ = self.renderer.render(scene_cam, scene_verts, self.texs.clone())
        img = img[0]
        # img = self.renderer.render_silhouettes(scene_cam, scene_verts)
        img = img.permute(1, 2, 0)
        img = img.cpu().detach().numpy()

        # Normalize:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img + 1) / 2.0 * 255
        img = img.astype(np.uint8)

        if resize_to is not None:
            img = cv2.resize(img, resize_to)

        return img


import os
import torch
import numpy as np
import cv2
import argparse
from vibert.lib.models.vibe_rt import VibeRT
from vibert.lib.data_utils.img_utils import get_single_image_crop_demo


class HoloVibeRT():
    def __init__(self, args):
        self.bbox_scale = args.bbox_scale
        self.crop_size = args.crop_size
        self.include_verts = args.include_verts
        self.device = torch.device('cuda:' + str(args.gpu_id))

        # Init VIBE-RT:
        self.vibe_model = VibeRT(seqlen=args.seqlen,
                                 n_layers=2, hidden_size=1024, add_linear=True, use_residual=True,
                                 pretrained=args.spin_model_path,
                                 smpl_model_dir=args.smpl_model_dir,
                                 smpl_mean_path=args.smpl_mean_path,
                                 joint_regressor_path=args.j_regressor_path)
        self.vibe_model = self.vibe_model.to(self.device)

        # Load pretrained VIBE model:
        ckpt = torch.load(args.vibe_model_path)
        self.vibe_model.load_state_dict(ckpt['gen_state_dict'], strict=False)
        self.vibe_model.eval()
        #print('Performance of pretrained VIBE model on 3DPW:', ckpt["performance"])

    def inference(self, frame):
        with torch.no_grad():
            while frame.ndimension() < 5:
                frame = frame.unsqueeze(0)
            frame = frame.to(self.device)

            preds = self.vibe_model(frame)[-1]

            output = {
                'pred_cam': preds['theta'][:, :, :3].reshape(1, -1).cpu(),
                'pose': preds['theta'][:, :, 3:75].reshape(1, -1).cpu(),
                'betas': preds['theta'][:, :, 75:].reshape(1, -1).cpu(),
                'rotmat': preds['rotmat'].reshape(1, -1, 3, 3).cpu()
            }
            if self.include_verts:
                output['verts'] = preds['verts'].reshape(1, -1, 3).cpu()

            return output


def convert_cam(cam, bbox1, bbox2, truncated=True):
    bbox = bbox1[:,:3] - bbox2[:,:3]
    bbox[:,2] = bbox1[:,2]
    img_width = bbox2[0,2]
    img_height = bbox2[0,3]

    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx ) / hw / sx) + cam[:,1]
    ty = ((cy ) / hh / sy) + cam[:,2]

    if truncated:
        orig_cam = np.stack([sx, tx, ty]).T
    else:
        orig_cam = np.stack([sx, sy, tx, ty]).T

    return orig_cam


def init_vibe(vibe_conf):
    # Init params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='repo root dir')
    parser.add_argument('--smpl_model_dir', type=str,
                        default='',
                        help='dir the containing SMPL model')
    parser.add_argument('--smpl_mean_path', type=str,
                        default='smpl_mean_params.npz',
                        help='path to SMPL mean params file')
    parser.add_argument('--j_regressor_path', type=str,
                        default='J_regressor_extra.npy',
                        help='path to Joint regressor model')
    parser.add_argument('--spin_model_path', type=str,
                        default='spin_model_checkpoint.pth.tar',
                        help='path to spin model')
    parser.add_argument('--vibe_model_path', type=str,
                        default='vibe_model_wo_3dpw.pth.tar',
                        help='path to pretrained VIBE model')
    parser.add_argument('--seqlen', type=int, default=16,
                        help='VIBE sequence length')
    parser.add_argument('--bbox_scale', type=float, default=1.1,
                        help='scale for input bounding box')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size for input image')
    parser.add_argument('--include_verts', action='store_true', default=False,
                        help='include smpl vertices to a vibe output')
    args, unknown = parser.parse_known_args()

    # Set params:
    args.seqlen = vibe_conf['seqlen']
    args.root_dir = vibe_conf['root_dir']
    args.spin_model_path = os.path.join(args.root_dir, args.spin_model_path)
    args.smpl_model_dir = os.path.join(args.root_dir, args.smpl_model_dir)
    args.smpl_mean_path = os.path.join(args.root_dir, args.smpl_mean_path)
    args.j_regressor_path = os.path.join(args.root_dir, args.j_regressor_path)
    args.vibe_model_path = os.path.join(args.root_dir, args.vibe_model_path)
    args.gpu_id = vibe_conf['gpu_id']
    if 'include_verts' in vibe_conf:
        args.include_verts = vibe_conf['include_verts']

    # Init vibe:
    print('Initializing VIBE...')
    vibe = HoloVibeRT(args)
    print('Loaded pretrained VIBE weights from', args.vibe_model_path)

    return vibe, args


def pre_vibe(data, args):
    # Prepare VIBE input:
    frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
    norm_img, _, _ = get_single_image_crop_demo(frame_rgb, data['yolo_cbbox'][0], kp_2d=None,
                                                scale=args.bbox_scale, crop_size=args.crop_size)
    data['vibe_input'] = norm_img.unsqueeze(0)

    return data


def post_vibe(data):
    # Prepare VIBE output:
    scene_cam = convert_cam(cam=data['vibe_output']['pred_cam'].numpy(),
                            bbox1=data['yolo_cbbox'],
                            bbox2=data['scene_cbbox'],
                            truncated=False)
    avatar_cam = np.stack([[scene_cam[0,0], scene_cam[0,2], scene_cam[0,3]]])
    smpl_vec = np.concatenate((avatar_cam,
                               data['vibe_output']['pose'].numpy(),
                               data['vibe_output']['betas'].numpy()), axis=1)
    data['scene_cam'] = scene_cam
    data['smpl_vec'] = smpl_vec

    return data
import os
import argparse
import torch
import numpy as np
import cv2
from vibert.lib.models.vibe_rt import VibeRT
from vibert.lib.data_utils.img_utils import get_single_image_crop_demo
from vibert.holo.data_struct import DataStruct
import time


class HoloVibeRT():
    def __init__(self, args):
        # Set params:
        spin_model_path = os.path.join(args.root_dir, args.spin_model_path)
        smpl_model_dir = os.path.join(args.root_dir, args.smpl_model_dir)
        smpl_mean_path = os.path.join(args.root_dir, args.smpl_mean_path)
        j_regressor_path = os.path.join(args.root_dir, args.j_regressor_path)
        vibe_model_path = os.path.join(args.root_dir, args.vibe_model_path)
        self.bbox_scale = args.bbox_scale
        self.crop_size = args.crop_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Init VIBE-RT:
        self.vibe_model = VibeRT(seqlen=args.seqlen,
                                 n_layers=2, hidden_size=1024, add_linear=True, use_residual=True,
                                 pretrained=spin_model_path,
                                 smpl_model_dir=smpl_model_dir,
                                 smpl_mean_path=smpl_mean_path,
                                 joint_regressor_path=j_regressor_path)
        self.vibe_model = self.vibe_model.to(self.device)

        # Load pretrained VIBE model:
        ckpt = torch.load(vibe_model_path)
        self.vibe_model.load_state_dict(ckpt['gen_state_dict'], strict=False)
        self.vibe_model.eval()
        print('Loaded pretrained VIBE weights from', vibe_model_path)
        print('Performance of pretrained model on 3DPW:', ckpt["performance"])

    def inference(self, frame):
        with torch.no_grad():
            frame = frame.unsqueeze(0).unsqueeze(0)
            frame = frame.to(self.device)

            preds = self.vibe_model(frame)[-1]

            return {
                'pred_cam': preds['theta'][:, :, :3].reshape(1, -1).cpu(),
                'pose': preds['theta'][:, :, 3:75].reshape(1, -1).cpu(),
                'betas': preds['theta'][:, :, 75:].reshape(1, -1).cpu(),
                'rotmat': preds['rotmat'].reshape(1, -1, 3, 3).cpu()
            }


def convert_crop_cam_to_another_crop(cam, bbox1, bbox2, img_width, img_height):
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
    orig_cam = np.stack([sx, sy, tx, ty]).T

    return orig_cam


def load_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, scale=1.1, crop_size=224):
    data_frames = DataStruct().parse(frames_dir, levels='subject/light/garment/scene/cam', ext='jpeg')
    data_yolo_bboxes = DataStruct().parse(yolo_bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')
    data_avatar_bbox = DataStruct().parse(avatar_bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')
    frames = []
    bboxes= []

    for (f_node, f_path), (y_node, y_path), (a_node, a_path) in \
            zip(data_frames.nodes('cam'), data_yolo_bboxes.nodes('cam'), data_avatar_bbox.nodes('cam')):
        if f_path != target_path:
            continue
        print ('Processing dir', f_path)
        assert f_path == y_path and f_path == a_path

        # Unpack npz containing yolo bboxes:
        bboxes_path = [npz.abs_path for npz in data_yolo_bboxes.items(y_node)][0]
        bboxes_npz = np.load(bboxes_path, encoding='latin1', allow_pickle=True)
        frame_ids = np.array(bboxes_npz['frames'])
        yolo_bboxes = np.array(bboxes_npz['bboxes'])
        yolo_bboxes[:, 0] = yolo_bboxes[:, 0] + yolo_bboxes[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        yolo_bboxes[:, 1] = yolo_bboxes[:, 1] + yolo_bboxes[:, 3] * 0.5

        # Unpack npz containing avatar bboxes:
        bboxes_path = [npz.abs_path for npz in data_avatar_bbox.items(a_node)][0]
        bboxes_npz = np.load(bboxes_path, encoding='latin1', allow_pickle=True)
        avatar_bboxes = np.array(bboxes_npz['bboxes'])
        avatar_bboxes[:, 0] = avatar_bboxes[:, 0] + avatar_bboxes[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        avatar_bboxes[:, 1] = avatar_bboxes[:, 1] + avatar_bboxes[:, 3] * 0.5

        # Prepare frames:
        frame_paths = np.array([f.path for f in data_frames.items(f_node)])
        frame_paths = frame_paths[frame_ids]
        assert len(frame_paths) == yolo_bboxes.shape[0]

        for i in range(len(frame_paths)):
            img_path = os.path.join(frames_dir, frame_paths[i])
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            norm_img,_,_ = get_single_image_crop_demo(img, yolo_bboxes[i], kp_2d=None, scale=scale, crop_size=crop_size)
            frames.append(norm_img.unsqueeze(0))
        break

    return torch.cat(frames, dim=0), yolo_bboxes, avatar_bboxes, frame_paths


def main():
    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='repo root dir')
    parser.add_argument('--smpl_model_dir', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data',
                        help='dir the containing SMPL model')
    parser.add_argument('--smpl_mean_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/smpl_mean_params.npz',
                        help='path to SMPL mean params file')
    parser.add_argument('--j_regressor_path', type=str,
                       default='thirdparty/vibe/vibert/data/vibe_data/J_regressor_extra.npy',
                       help='path to Joint regressor model')
    parser.add_argument('--spin_model_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/spin_model_checkpoint.pth.tar',
                        help='path to spin model')
    parser.add_argument('--vibe_model_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/vibe_model_wo_3dpw.pth.tar',
                        help='path to pretrained VIBE model')
    parser.add_argument('--seqlen', type=int, default=16,
                        help='VIBE sequence length')
    parser.add_argument('--bbox_scale', type=float, default=1.1,
                        help='scale for input bounding box')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size for input image')
    args = parser.parse_args()

    args.seqlen = 1

    # Init VIBE-RT model:
    print ('Initializing VIBE...')
    vibe = HoloVibeRT(args)

    # Load test data:
    print('Loading test data...')
    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames'
    yolo_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes_by_maskrcnn'
    avatar_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes'
    target_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'
    frames, yolo_bboxes, avatar_bboxes, frame_paths = \
        load_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, args.bbox_scale, args.crop_size)
    print('Test data has been loaded:', frames.shape)

    # Inference:
    print('Inferencing...')
    pred_cam, pred_pose, pred_betas, pred_rotmat = [], [], [], []
    save_output = True
    start = time.time()

    for frame in frames:
        output = vibe.inference(frame)

        if save_output:
            pred_cam.append(output['pred_cam'])
            pred_pose.append(output['pose'])
            pred_betas.append(output['betas'])
            pred_rotmat.append(output['rotmat'])

    elapsed = time.time() - start
    fps = frames.shape[0] / elapsed
    print('Elapsed time:', elapsed, 'frames:', frames.shape[0], 'fps:', fps)

    if save_output:
        pred_cam = torch.cat(pred_cam, dim=0).numpy()
        pred_pose = torch.cat(pred_pose, dim=0).numpy()
        pred_betas = torch.cat(pred_betas, dim=0).numpy()
        pred_rotmat = torch.cat(pred_rotmat, dim=0).numpy()

        # Make dummy bboxes:
        img_width, img_height = 1920, 1080

        # Recalculate cam params:
        avatar_cam = convert_crop_cam_to_another_crop(cam=pred_cam,
                                                      bbox1=yolo_bboxes,
                                                      bbox2=avatar_bboxes,
                                                      img_width=img_width, img_height=img_height)
        if avatar_cam.shape[1] > 3:
            avatar_cam = np.stack((avatar_cam[:, 0], avatar_cam[:, 2], avatar_cam[:, 3]), axis=1)
        assert avatar_cam.shape[1] == 3

        # Save the result:
        result_dir = os.path.join('/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpls_by_vibe-rt',target_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'smpl.npz')
        np.savez(result_path,
                 avatar_cam=avatar_cam,
                 pose=pred_pose,
                 betas=pred_betas,
                 rotmat=pred_rotmat,
                 frame_paths=frame_paths)

    print ('All done!')



if __name__ == '__main__':
    main()
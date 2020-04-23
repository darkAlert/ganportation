import sys
import os
import numpy as np
import cv2
import time
from vibert.lib.data_utils.img_utils import get_single_image_crop_demo
from vibert.holo.data_struct import DataStruct
from holoport.hvibe import init_vibe, convert_cam
from holoport.conf.conf_parser import parse_conf


def load_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, scale=1.1, crop_size=224):
    data_frames = DataStruct().parse(frames_dir, levels='subject/light/garment/scene/cam', ext='jpeg')
    data_yolo_bboxes = DataStruct().parse(yolo_bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')
    data_avatar_bbox = DataStruct().parse(avatar_bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')
    frames, vibe_frames = [], []
    yolo_cbboxes, avatar_bboxes, avatar_cbboxes, frame_paths = None, None, None, None

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
        yolo_cbboxes = np.array(bboxes_npz['bboxes'])
        yolo_cbboxes[:, 0] = yolo_cbboxes[:, 0] + yolo_cbboxes[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        yolo_cbboxes[:, 1] = yolo_cbboxes[:, 1] + yolo_cbboxes[:, 3] * 0.5

        # Unpack npz containing avatar bboxes:
        bboxes_path = [npz.abs_path for npz in data_avatar_bbox.items(a_node)][0]
        bboxes_npz = np.load(bboxes_path, encoding='latin1', allow_pickle=True)
        avatar_bboxes = np.array(bboxes_npz['bboxes'])
        avatar_cbboxes = avatar_bboxes.copy()
        avatar_cbboxes[:, 0] = avatar_cbboxes[:, 0] + avatar_cbboxes[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        avatar_cbboxes[:, 1] = avatar_cbboxes[:, 1] + avatar_cbboxes[:, 3] * 0.5

        # Prepare frames:
        frame_paths = np.array([f.path for f in data_frames.items(f_node)])
        frame_paths = frame_paths[frame_ids]
        assert len(frame_paths) == yolo_cbboxes.shape[0]

        for i in range(len(frame_paths)):
            img_path = os.path.join(frames_dir, frame_paths[i])
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            frames.append(img)
            norm_img,_,_ = get_single_image_crop_demo(img, yolo_cbboxes[i], kp_2d=None, scale=scale, crop_size=crop_size)
            vibe_frames.append(norm_img.unsqueeze(0))
        break

    assert len(frames) == len(vibe_frames) and len(frames) == yolo_cbboxes.shape[0] and len(frames) == avatar_bboxes.shape[0] and len(frames) == frame_paths.shape[0]

    # Pack data:
    data = []
    for i in range(len(frames)):
        data.append({
            'frame': frames[i],
            'vibe_input': vibe_frames[i],
            'yolo_cbbox': np.expand_dims(yolo_cbboxes[i], axis=0),
            'scene_bbox' : np.expand_dims(avatar_bboxes[i], axis=0),
            'scene_cbbox': np.expand_dims(avatar_cbboxes[i], axis=0),
            'path' : np.expand_dims(frame_paths[i], axis=0)
        })

    return data


def test(path_to_conf, save_results=False):
    # Load config:
    conf = parse_conf(path_to_conf)
    print ('Config has been loaded from', path_to_conf)

    # Init VIBE-RT model:
    conf['vibe']['gpu_id'] = '0'
    vibe, args = init_vibe(conf['vibe'])

    # Load test data:
    print('Loading test data...')
    frames_dir = conf['input']['frames_dir']
    yolo_bboxes_dir = conf['input']['yolo_bboxes_dir']
    avatar_bboxes_dir = conf['input']['avatar_bboxes_dir']
    target_path = conf['input']['target_path']
    test_data = load_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, args.bbox_scale, args.crop_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    start = time.time()

    for data in test_data:
        output = vibe.inference(data['vibe_input'])

        avatar_cam = convert_cam(cam=output['pred_cam'].numpy(),
                                 bbox1=data['yolo_cbbox'],
                                 bbox2=data['scene_cbbox'],
                                 truncated=True)
        data['smpl'] = {
            'pred_cam': output['pred_cam'].numpy(),
            'pose': output['pose'].numpy(),
            'betas': output['betas'].numpy(),
            'rotmat': output['rotmat'].numpy(),
            'avatar_cam': avatar_cam,
        }

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    spf = elapsed / len(test_data)  # secons per frame
    print('###Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps, 'spf:', spf)

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        pose, betas, rotmat, avatar_cam, frame_paths = [], [], [], [], []

        # Merge outputs:
        for data in test_data:
            pose.append(data['smpl']['pose'])
            betas.append(data['smpl']['betas'])
            rotmat.append(data['smpl']['rotmat'])
            avatar_cam.append(data['smpl']['avatar_cam'])
            frame_paths.append(data['path'])
        pose = np.concatenate(pose, axis=0)
        betas = np.concatenate(betas, axis=0)
        rotmat = np.concatenate(rotmat, axis=0)
        avatar_cam = np.concatenate(avatar_cam, axis=0)
        frame_paths = np.concatenate(frame_paths, axis=0)

        # Save:
        output_dir = os.path.join(result_dir,target_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'smpl.npz')

        np.savez(output_path,
                 avatar_cam=avatar_cam,
                 pose=pose,
                 betas=betas,
                 rotmat=rotmat,
                 frame_paths=frame_paths)
        print ('The results have been saved to', result_dir)

    print ('All done!')


if __name__ == '__main__':
    path_to_conf = 'holoport/conf/local/vibe_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    test(path_to_conf, save_results=True)

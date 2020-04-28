import sys
import os
import time
import numpy as np
import cv2
from tqdm import tqdm

from holoport.conf.conf_parser import parse_conf
from holoport.tests.test_yolo import test as test_yolo
from holoport.tests.test_vibe import test as test_vibe
from holoport.tests.test_lwgan import test as test_lwgan
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.hvibe import init_vibe, convert_cam
from holoport.hyolo import init_yolo, prepare_yolo_input, convert_yolo_output_to_bboxes
from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input
from vibert.lib.data_utils.img_utils import get_single_image_crop_demo


def get_file_paths(path, exts=('.jpeg','.jpg','.png')):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if any(f.endswith(ext) for ext in exts)]:
            file_paths.append(os.path.join(dirpath,filename))
            file_paths.sort()

    return file_paths


def load_frames(frames_dir, max_frames=None):
    frames = []
    frame_paths = get_file_paths(frames_dir)

    if max_frames is None:
        max_frames = len(frame_paths)
    if max_frames > len(frame_paths):
        max_frames = len(frame_paths)

    for idx, path in enumerate(frame_paths):
        if idx >= max_frames:
            break
        frames.append({'frame': cv2.imread(path,1)})

    return frames


def pre_yolo(data, args):
    # Prepare YOLO input:
    data['yolo_input'] = prepare_yolo_input(data['frame'].copy(), args.yolo_img_size)

    return data


def post_yolo(data):
    # Prepare YOLO output:
    actual_size = [data['yolo_input'].shape[2:]]
    origin_size = [data['frame'].shape]
    bboxes = convert_yolo_output_to_bboxes(data['yolo_output'], actual_size, origin_size)

    # Covert predicted bbox:
    if len(bboxes):
        # (x1,y1,x2,y2) -> (x1,y1,w,h)
        bbox = bboxes[0].clone().numpy()
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        # (x1,y1,w,h) -> (x,y,s,s), s=max(w,h), x=x1-dx, y=y1-dy
        side = max(bbox[2], bbox[3])
        dx = (side - bbox[2]) * 0.5
        dy = (side - bbox[3]) * 0.5
        bbox[0] = bbox[0] - dx
        bbox[1] = bbox[1] - dy
        bbox[2] = side
        bbox[3] = side
        data['yolo_bbox'] = np.expand_dims(bbox, axis=0)

        # (x,y,w,h) -> (cx,cy,w,h)
        yolo_cbbox = np.copy(data['yolo_bbox'])
        yolo_cbbox[:,0] = yolo_cbbox[:,0] + yolo_cbbox[:,2] * 0.5
        yolo_cbbox[:,1] = yolo_cbbox[:,1] + yolo_cbbox[:,3] * 0.5
        data['yolo_cbbox'] = yolo_cbbox
    else:
        data['yolo_bbox'] = None
        data['yolo_cbbox'] = None

    return data


def pre_vibe(data, args):
    # Prepare VIBE input:
    frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
    norm_img, _, _ = get_single_image_crop_demo(frame_rgb, data['yolo_cbbox'][0], kp_2d=None,
                                                scale=args.bbox_scale, crop_size=args.crop_size)
    data['vibe_input'] = norm_img.unsqueeze(0)

    return data


def post_vibe(data):
    # Prepare VIBE output:
    avatar_cam = convert_cam(cam=data['vibe_output']['pred_cam'].numpy(),
                             bbox1=data['yolo_cbbox'],
                             bbox2=data['scene_cbbox'],
                             truncated=True)
    smpl_vec = np.concatenate((avatar_cam,
                               data['vibe_output']['pose'].numpy(),
                               data['vibe_output']['betas'].numpy()), axis=1)
    data['smpl_vec'] = smpl_vec

    return data


def pre_lwgan(data, args):
    # Prepare LWGAN input:
    frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
    bbox = data['scene_bbox'][0]
    scene_img = frame_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    prep_img, prep_smpl = prepare_lwgan_input(scene_img, data['smpl_vec'], args.image_size)
    data['lwgan_input_img'] = prep_img
    data['lwgan_input_smpl'] = prep_smpl

    return data


def post_lwgan(data):
    # Prepare LWGAN output:
    avatar = cv2.cvtColor(data['lwgan_output'], cv2.COLOR_RGB2BGR)

    # Normalize avatar:
    avatar = (avatar + 1) / 2.0 * 255
    data['avatar'] = avatar.astype(np.uint8)

    return data


def test_yolo_vibe_lwgan(path_to_conf, save_results=True):
    # Load configs:
    conf = parse_conf(path_to_conf)
    print('Config has been loaded from', path_to_conf)

    # Init YOLO-RT model:
    conf['yolo']['gpu_id'] = '0'
    yolo, yolo_args = init_yolo(conf['yolo'])

    # Init VIBE-RT model:
    conf['vibe']['gpu_id'] = '0'
    vibe, vibe_args = init_vibe(conf['vibe'])

    # Init LWGAN-RT model:
    conf['lwgan']['gpu_ids'] = '1'
    lwgan, lwgan_args = init_lwgan(conf['lwgan'])

    # Load test data:
    print('Loading test data...')
    frames_dir = os.path.join(conf['input']['frames_dir'], conf['input']['target_path'])
    n = int(conf['input']['max_frames']) if 'max_frames' in conf['input'] else None
    test_data = load_frames(frames_dir, max_frames=n)
    print('Test data has been loaded:', len(test_data))

    # Avatar view params:
    steps = conf['input']['steps']
    view = parse_view_params(conf['input']['view'])
    delta = 360 / steps
    step_i = 0

    # Dummy scene params:
    t = conf['input']['scene_bbox'].split(',')
    assert len(t) == 4
    dummy_scene_bbox = np.array([[int(t[0]), int(t[1]), int(t[2]), int(t[3])]], dtype=np.int64)
    dummy_scene_cbbox = dummy_scene_bbox.copy()
    dummy_scene_cbbox[:,0] = dummy_scene_bbox[:,0] + dummy_scene_bbox[:,2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
    dummy_scene_cbbox[:,1] = dummy_scene_bbox[:,1] + dummy_scene_bbox[:,3] * 0.5

    # Inference:
    print('Inferencing...')
    start = time.time()
    pre_yolo_elapsed, post_yolo_elapsed = 0, 0
    pre_vibe_elapsed, post_vibe_elapsed = 0, 0
    pre_lwgan_elapsed, post_lwgan_elapsed = 0, 0

    for idx, data in enumerate(tqdm(test_data)):
        # Update avatar view:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0
        data['lwgan_input_view'] = view

        # YOLO:
        t_start = time.time()
        data = pre_yolo(data, yolo_args)
        pre_yolo_elapsed += time.time() - t_start
        data['yolo_output'] = yolo.inference(data['yolo_input'])
        t_start = time.time()
        data = post_yolo(data)
        post_yolo_elapsed += time.time() - t_start

        if data['yolo_cbbox'] is None:
            print ('Skip frame {}: person not found!'.format(idx))
            continue

        # Scene bbox and cbbox:
        data['scene_bbox'] = dummy_scene_bbox
        data['scene_cbbox'] = dummy_scene_cbbox

        # VIBE:
        t_start = time.time()
        data = pre_vibe(data, vibe_args)
        pre_vibe_elapsed += time.time() - t_start
        data['vibe_output'] = vibe.inference(data['vibe_input'])
        t_start = time.time()
        data = post_vibe(data)
        post_vibe_elapsed += time.time() - t_start

        # LWGAN:
        t_start = time.time()
        data = pre_lwgan(data, lwgan_args)
        pre_lwgan_elapsed += time.time() - t_start
        data['lwgan_output'] = lwgan.inference(data['lwgan_input_img'],
                                               data['lwgan_input_smpl'],
                                               data['lwgan_input_view'])
        t_start = time.time()
        data = post_lwgan(data)
        post_lwgan_elapsed += time.time() - t_start

        step_i += 1
        if step_i >= steps:
            step_i = 0

    elapsed = time.time() - start
    n = len(test_data)
    fps = n / elapsed
    spf = elapsed / len(test_data)  # seconds per frame
    print('###Elapsed time:', elapsed, 'frames:', n, 'fps:', fps, 'spf:', spf)
    print('Mean pre yolo:', pre_yolo_elapsed / n, ', post yolo:', post_yolo_elapsed / n)
    print('Mean pre vibe:', pre_vibe_elapsed / n, ', post vibe:', post_vibe_elapsed / n)
    print('Mean pre lwgan:', pre_lwgan_elapsed / n, ', post lwgan:', post_lwgan_elapsed / n)

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for idx, data in enumerate(test_data):
            out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
            cv2.imwrite(out_path, data['avatar'])


def main():
    path_to_conf = 'holoport/conf/local/yolo_vibe_lwgan_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    if False:
        print ('==============Testing YOLO==============')
        test_yolo('holoport/conf/local/yolo_conf_local.yaml', False)
        print('==============Testing VIBE==============')
        test_vibe('holoport/conf/local/vibe_conf_local.yaml', False)
        print('==============Testing LWGAN==============')
        test_lwgan('holoport/conf/local/lwgan_conf_local.yaml', False)

    test_yolo_vibe_lwgan(path_to_conf)

if __name__ == '__main__':
    main()
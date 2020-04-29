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
from holoport.hlwgan import init_lwgan, pre_lwgan, post_lwgan, parse_view_params
from holoport.hvibe import init_vibe, pre_vibe, post_vibe
from holoport.hyolo import init_yolo, pre_yolo, post_yolo
from holoport.workers import warmup_holoport_pipeline


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


def test(path_to_conf, save_results=True):
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

    # Warmup:
    if 'warmup_img' in conf['input']:
        print('Warming up holoport pipeline...')
        img = cv2.imread(conf['input']['warmup_img'],1)
        warmup_holoport_pipeline(img, yolo, yolo_args, vibe, vibe_args, lwgan, lwgan_args)

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

    test(path_to_conf)

if __name__ == '__main__':
    main()
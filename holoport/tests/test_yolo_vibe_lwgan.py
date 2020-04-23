import os
import time
import numpy as np

from holoport.tests.test_yolo import test as test_yolo
from holoport.tests.test_vibe import test as test_vibe
from holoport.tests.test_lwgan import test as test_lwgan


from holoport.tests.test_lwgan import init_lwgan
from holoport.tests.test_yolo import load_data as load_yolo_data
from holoport.tests.test_vibe import load_data as load_vibe_data

from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input
from lwganrt.utils.cv_utils import save_cv2_img

from holoport.hvibe import init_vibe, convert_cam
from holoport.hlwgan import parse_view_params
from holoport.conf.conf_parser import parse_conf

def prepare_vibe_test_data(conf_input, conf_vibe):
    frames_dir = conf_input['frames_dir']
    yolo_bboxes_dir = conf_input['yolo_bboxes_dir']
    avatar_bboxes_dir = conf_input['avatar_bboxes_dir']
    target_path = conf_input['target_path']
    bbox_scale = conf_vibe['scale']
    crop_size = conf_vibe['crop_size']

    test_data = load_vibe_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, bbox_scale, crop_size)

    return test_data


def test_vibe_lwgan(path_to_conf):
    # Load configs:
    conf = parse_conf(path_to_conf)
    print('Config has been loaded from', path_to_conf)

    # Init VIBE-RT model:
    conf['vibe']['gpu_id'] = '0'
    vibe, vibe_args = init_vibe(conf['vibe'])

    # Init LWGAN-RT model:
    conf['lwgan']['gpu_ids'] = '1'
    lwgan, lwgan_args = init_lwgan(conf['lwgan'])

    # Load test data:
    print('Loading test data...')
    test_data = prepare_vibe_test_data(conf['input'], conf['vibe'])
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    result_dir = conf['output']['result_dir']
    steps = conf['input']['steps']
    view = parse_view_params(conf['input']['view'])
    delta = 360 / steps
    step_i = 0
    start = time.time()

    for data in test_data:
        # Update view::
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0
        step_i += 1
        if step_i >= steps:
            step_i = 0

        # VIBE:
        vibe_out = vibe.inference(data['vibe_input'])

        # Prepare input for LWGAN:
        avatar_cam = convert_cam(cam=vibe_out['pred_cam'].numpy(),
                                 bbox1=data['yolo_cbbox'],
                                 bbox2=data['scene_cbbox'],
                                 truncated=True)
        smpl_vec = np.concatenate((avatar_cam,
                                   vibe_out['pose'].numpy(),
                                   vibe_out['betas'].numpy()), axis=1)
        # Crop:
        bbox = data['scene_bbox'][0]
        scene_img = data['frame'][bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        prep_img, prep_smpl = prepare_lwgan_input(scene_img, smpl_vec, lwgan_args.image_size)
        data['lwgan_input'] = prep_img
        data['smpl'] = prep_smpl
        data['view'] = view

        # LWGAN:
        lwgan_out = lwgan.inference(data['lwgan_input'], data['smpl'], data['view'])

        # Pack the result:
        data['avatar'] = lwgan_out

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps)

    # Save the results:
    if result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for idx, data in enumerate(test_data):
            out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
            save_cv2_img(data['avatar'], out_path, normalize=True)


def main():
    if True:
        print ('==============Testing YOLO==============')
        test_yolo('holoport/conf/local/yolo_conf_local.yaml', False)
        print('==============Testing VIBE==============')
        test_vibe('holoport/conf/local/vibe_conf_local.yaml', False)
        print('==============Testing LWGAN==============')
        test_lwgan('holoport/conf/local/lwgan_conf_local.yaml', False)

    # test_vibe_lwgan(path_to_conf='holoport/conf/local/vibe_lwgan_conf_local.yaml')

if __name__ == '__main__':
    main()
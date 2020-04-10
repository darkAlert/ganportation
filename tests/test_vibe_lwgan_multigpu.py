import os
import time
import numpy as np
import cv2
from tests.test_vibe import load_data as load_vibe_data
from tests.test_vibe import init_vibe
from tests.test_lwgan import load_data as load_lwgan_data
from tests.test_lwgan import init_lwgan
from hvibe import convert_cam
from hlwgan import parse_view_params
from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input
from lwganrt.utils.cv_utils import save_cv2_img


def prepare_vibe_test_data(bbox_scale, crop_size):
    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames'
    yolo_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes_by_maskrcnn'
    avatar_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes'
    target_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'

    test_data = load_vibe_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, bbox_scale, crop_size)

    return test_data


def prepare_lwgan_test_data(image_size):
    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/avatars'
    smpls_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpls_by_vibe_aligned_lwgan'
    scene_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'
    frames_dir = os.path.join(frames_dir, scene_path)
    smpls_dir = os.path.join(smpls_dir, scene_path)

    return load_lwgan_data(frames_dir, smpls_dir, image_size)


def test_vibe():
    # Init VIBE-RT model:
    vibe, args = init_vibe()

    # Load test data:
    print('Loading vibe test data...')
    test_data = prepare_vibe_test_data(args.bbox_scale, args.crop_size)
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
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps)


def test_lwgan(steps = 120):
    # Init LWGAN-RT model:
    lwgan, args = init_lwgan()

    # Load test data:
    print('Loading lwgan test data...')
    test_data = prepare_lwgan_test_data(args.image_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    view = parse_view_params('R=0,90,0/t=0,0,0')
    delta = 360 / steps
    step_i = 0
    lwgan_outputs = []
    start = time.time()

    for data in test_data:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0

        step_i += 1
        if step_i >= steps:
            step_i = 0

        preds = lwgan.inference(data['lwgan_input'], data['smpl'], view)
        lwgan_outputs.append(preds)

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps)


def test_vibe_lwgan(steps = 120):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    result_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/rt/t2'

    # Init VIBE-RT model:
    vibe, vibe_args = init_vibe()

    # Init LWGAN-RT model:
    lwgan, lwgan_args = init_lwgan()

    # Load test data:
    print('Loading test data...')
    test_data = prepare_vibe_test_data(vibe_args.bbox_scale, vibe_args.crop_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    view = parse_view_params('R=0,90,0/t=0,0,0')
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
    # test_vibe()
    # test_lwgan()
    test_vibe_lwgan()

    print('All done!')


if __name__ == '__main__':
    main()
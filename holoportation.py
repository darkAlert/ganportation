import os
import time
import numpy as np
from tests.test_vibe import load_data as load_vibe_data
from tests.test_vibe import init_vibe
from tests.test_lwgan import load_data as load_lwgan_data
from tests.test_lwgan import init_lwgan
from hlwgan import parse_view_params


def prepare_vibe_test_data(bbox_scale, crop_size):
    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames'
    yolo_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes_by_maskrcnn'
    avatar_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes'
    target_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'

    frames, yolo_bboxes, avatar_bboxes, frame_paths = \
        load_vibe_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, bbox_scale, crop_size)

    return frames, yolo_bboxes, avatar_bboxes, frame_paths


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
    frames, yolo_bboxes, avatar_bboxes, frame_paths = prepare_vibe_test_data(args.bbox_scale, args.crop_size)
    print('Test data has been loaded:', frames.shape)

    # Inference:
    print('Inferencing...')
    vibe_outputs = []
    start = time.time()

    for frame in frames:
        output = vibe.inference(frame)
        vibe_outputs.append(output)

    elapsed = time.time() - start
    fps = frames.shape[0] / elapsed
    print('Elapsed time:', elapsed, 'frames:', frames.shape[0], 'fps:', fps)


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

    for sample in test_data:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0

        step_i += 1
        if step_i >= steps:
            step_i = 0

        preds = lwgan.inference(sample['frame'], sample['smpl'], view)
        lwgan_outputs.append(preds)

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps)


def main():
    test_vibe()
    test_lwgan()

    print('All done!')


if __name__ == '__main__':
    main()
import os
import numpy as np
import cv2
from lwganrt.models.holoportator_rt import HoloportatorRT
from lwganrt.options.test_options import TestOptions
from lwganrt.utils.cv_utils import save_cv2_img
from hlwgan import HoloLwganRT, parse_view_params
import time


def get_file_paths(path, exts=('.jpeg','.jpg','.png')):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if any(f.endswith(ext) for ext in exts)]:
            file_paths.append(os.path.join(dirpath,filename))
            file_paths.sort()

    return file_paths


def load_data(frames_dir, smpls_dir, img_size):
    # Load frames:
    frame_paths = get_file_paths(frames_dir)
    images = []
    for path in frame_paths:
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    # Load smpls:
    smpl_path = get_file_paths(smpls_dir, exts=('.npz'))[0]
    smpls = []
    with np.load(smpl_path, encoding='latin1', allow_pickle=True) as data:
        smpl_data = dict(data)
        n = len(smpl_data['cams'])

        for frame_id in range(n):
            cams = smpl_data['cams'][frame_id]
            pose = smpl_data['pose'][frame_id]
            shape = smpl_data['shape'][frame_id]
            vec = np.concatenate((cams, pose, shape), axis=0)
            smpls.append(vec)

    assert len(images) == len(smpls)

    # Prepare data:
    data = []
    for img, smpl in zip(images, smpls):
        prep_img, prep_smpl = HoloportatorRT.prepare_input(img, smpl, img_size)
        data.append({'frame': prep_img, 'smpl': prep_smpl})

    return data


def init_lwgan(gpu_id='0'):
    # Set params:
    args = TestOptions().parse()
    args.gpu = gpu_id
    args.gen_name = "holoportator"
    args.image_size = 256
    args.bg_ks = 11
    args.ft_ks = 3
    args.has_detector = False
    args.post_tune = False
    args.front_warp = False
    args.save_res = False
    args.n_threads_test = 4
    args.load_path = '/home/darkalert/builds/impersonator/outputs/Holo_iPER/net_epoch_20_id_G.pth'
    args.smpl_model = '/home/darkalert/builds/impersonator/assets/pretrains/smpl_model.pkl'
    args.hmr_model = '/home/darkalert/builds/impersonator/assets/pretrains/hmr_tf2pt.pth'
    args.smpl_faces = '/home/darkalert/builds/impersonator/assets/pretrains/smpl_faces.npy'
    args.uv_mapping = '/home/darkalert/builds/impersonator/assets/pretrains/mapper.txt'
    args.part_info = '/home/darkalert/builds/impersonator/assets/pretrains/smpl_part_info.json'
    args.front_info = '/home/darkalert/builds/impersonator/assets/pretrains/front_facial.json'
    args.head_info = '/home/darkalert/builds/impersonator/assets/pretrains/head.json'

    # Init LWGAN-RT model:
    print('Initializing LWGAN-RT...')
    lwgan = HoloLwganRT(args)

    return lwgan, args


def main():
    result_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/rt/t1'
    steps = 120

    # Init LWGAN-RT model:
    lwgan, args = init_lwgan()

    # Load test data:
    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/avatars'
    smpls_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpls_by_vibe_aligned_lwgan'
    scene_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'
    frames_dir = os.path.join(frames_dir, scene_path)
    smpls_dir = os.path.join(smpls_dir, scene_path)
    print('Loading test data...')
    test_data = load_data(frames_dir, smpls_dir, args.image_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    view = parse_view_params('R=0,90,0/t=0,0,0')
    delta = 360 / steps
    step_i = 0
    results = []
    start = time.time()

    for sample in test_data:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0

        step_i += 1
        if step_i >= steps:
            step_i = 0

        preds = lwgan.inference(sample['frame'], sample['smpl'], view)
        results.append(preds)

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps)

    if result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for idx, preds in enumerate(results):
            out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
            save_cv2_img(preds, out_path, normalize=True)

    print ('All done!')



if __name__ == '__main__':
    main()
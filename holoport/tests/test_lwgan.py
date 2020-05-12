import os
import sys
import numpy as np
import cv2
import time
from lwganrt.utils.cv_utils import save_cv2_img
from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.conf.conf_parser import parse_conf


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
        prep_img, prep_smpl = prepare_lwgan_input(img, smpl, img_size)
        data.append({'lwgan_input': prep_img, 'smpl': prep_smpl})

    return data


def test(path_to_conf, save_results=False):
    # Load config:
    conf = parse_conf(path_to_conf)
    print ('Config has been loaded from', path_to_conf)

    # Init LWGAN-RT model:
    lwgan, args = init_lwgan(conf['lwgan'])
    lwgan.mode = 'predefined'

    # Load test data:
    target_path = conf['input']['target_path']
    frames_dir = os.path.join(conf['input']['frames_dir'], target_path)
    smpls_dir = os.path.join(conf['input']['smpls_dir'], target_path)
    print('Loading test data...')
    test_data = load_data(frames_dir, smpls_dir, args.image_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    steps = conf['input']['steps']
    view = parse_view_params(conf['input']['view'])
    delta = 360 / steps
    step_i = 0
    results = []
    start = time.time()

    for data in test_data:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0

        step_i += 1
        if step_i >= steps:
            step_i = 0

        preds = lwgan.inference(data['lwgan_input'], data['smpl'], view)
        results.append(preds)

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    spf = elapsed / len(test_data)  # secons per frame
    print('###Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps, 'spf:', spf)

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for idx, preds in enumerate(results):
            out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
            save_cv2_img(preds, out_path, normalize=True)

    print ('All done!')


def test_batch(path_to_conf, batch_size=2, save_results=False):
    # Load config:
    conf = parse_conf(path_to_conf)
    print ('Config has been loaded from', path_to_conf)

    # Init LWGAN-RT model:
    lwgan, args = init_lwgan(conf['lwgan'])
    lwgan.mode = 'predefined'

    # Load test data:
    target_path = conf['input']['target_path']
    frames_dir = os.path.join(conf['input']['frames_dir'], target_path)
    smpls_dir = os.path.join(conf['input']['smpls_dir'], target_path)
    print('Loading test data...')
    test_data = load_data(frames_dir, smpls_dir, args.image_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    steps = conf['input']['steps']
    view = parse_view_params(conf['input']['view'])
    delta = 360 / steps
    step_i = 0
    results = []
    start = time.time()

    frame_batch = []
    smpl_batch = []
    view_batch = []

    for data in test_data:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0

        step_i += 1
        if step_i >= steps:
            step_i = 0

        frame_batch.append(data['lwgan_input'])
        smpl_batch.append(data['smpl'])
        view_batch.append(view)
        if len(frame_batch) < batch_size:
            continue

        preds = lwgan.inference_batch(frame_batch, smpl_batch)
        for i in range(len(frame_batch)):
            results.append(preds[i])

        frame_batch, smpl_batch, view_batch = [], [], []

    if len(frame_batch) > 0:
        preds = lwgan.inference_batch(frame_batch, smpl_batch)
        for i in range(len(frame_batch)):
            results.append(preds[i])

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    spf = elapsed / len(test_data)  # secons per frame
    print('###Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps, 'spf:', spf)

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for idx, preds in enumerate(results):
            out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
            save_cv2_img(preds, out_path, normalize=True)

    print ('All done!')

if __name__ == '__main__':
    path_to_conf = 'holoport/conf/local/lwgan_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]


    test_batch(path_to_conf, batch_size=2, save_results=True)
    # test(path_to_conf, save_results=False)
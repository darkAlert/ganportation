import os
import numpy as np
import cv2
from lwganrt.options.test_options import TestOptions
from lwganrt.utils.cv_utils import save_cv2_img
from hlwgan import HoloLwganRT, parse_view_params
from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input
from conf.conf_parser import parse_conf
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
        prep_img, prep_smpl = prepare_lwgan_input(img, smpl, img_size)
        data.append({'lwgan_input': prep_img, 'smpl': prep_smpl})

    return data


def init_lwgan(lwgan_conf):
    # Set params:
    args = TestOptions().parse()
    args.gpu = lwgan_conf['gpu']
    args.gen_name = lwgan_conf['gen_name']
    args.image_size = lwgan_conf['image_size']
    args.bg_ks = lwgan_conf['bg_ks']
    args.ft_ks = lwgan_conf['ft_ks']
    args.has_detector = lwgan_conf['has_detector']
    args.post_tune = lwgan_conf['post_tune']
    args.front_warp = lwgan_conf['front_warp']
    args.save_res = lwgan_conf['save_res']
    args.n_threads_test = lwgan_conf['n_threads_test']
    args.load_path = lwgan_conf['load_path']
    args.smpl_model = lwgan_conf['smpl_model']
    args.hmr_model = lwgan_conf['hmr_model']
    args.smpl_faces = lwgan_conf['smpl_faces']
    args.uv_mapping = lwgan_conf['uv_mapping']
    args.part_info = lwgan_conf['part_info']
    args.front_info = lwgan_conf['front_info']
    args.head_info = lwgan_conf['head_info']

    # Init LWGAN-RT model:
    print('Initializing LWGAN-RT...')
    lwgan = HoloLwganRT(args)

    return lwgan, args


def main(path_to_conf):
    # Load config:
    conf = parse_conf(path_to_conf)
    print ('Config has been loaded from', path_to_conf)

    # Init LWGAN-RT model:
    lwgan, args = init_lwgan(conf['lwgan'])

    # Load test data:
    target_path = conf['input']['target_path']
    frames_dir = os.path.join(conf['input']['frames_dir'], target_path)
    smpls_dir = os.path.join(conf['input']['smpls_dir'], target_path)
    print('Loading test data...')
    test_data = load_data(frames_dir, smpls_dir, args.image_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    result_dir = conf['output']['result_dir']
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
    main(path_to_conf='conf/local/lwgan_conf_local.yaml')
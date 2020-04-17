import torch
import numpy as np
from lwganrt.models.holoportator_rt import HoloportatorRT
from lwganrt.options.test_options import TestOptions


class HoloLwganRT:
    def __init__(self, args):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.holoport_model = HoloportatorRT(args, device=self.device)

    def inference(self, frame, smpl, view):
        frame = frame.to(self.device)
        smpl = smpl.to(self.device)

        # Personalize model:
        self.holoport_model.personalize(frame, smpl)

        # Inference:
        preds = self.holoport_model.view(view['R'], view['t'])

        return preds


def parse_view_params(view_params):
    """
    :param view_params: R=xxx,xxx,xxx/t=xxx,xxx,xxx
    :return:
        -R: np.ndarray, (3,)
        -t: np.ndarray, (3,)
    """
    params = dict()

    for segment in view_params.split('/'):
        # R=xxx,xxx,xxx -> (name, xxx,xxx,xxx)
        name, params_str = segment.split('=')
        vals = [float(val) for val in params_str.split(',')]
        params[name] = np.array(vals, dtype=np.float32)
    params['R'] = params['R'] / 180 * np.pi

    return params


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
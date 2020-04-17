import torch
import numpy as np
from lwganrt.models.holoportator_rt import HoloportatorRT
from lwganrt.options.test_options import TestOptions


class HoloLwganRT:
    def __init__(self, args):
        assert torch.cuda.is_available()
        self.device = torch.device('cuda:' + str(args.gpu_ids))
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
    args = TestOptions().parse(lwgan_conf, set_cuda_env=False)

    # Init LWGAN-RT model:
    print('Initializing LWGAN-RT...')
    lwgan = HoloLwganRT(args)

    return lwgan, args
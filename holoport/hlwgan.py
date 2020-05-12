import torch
import numpy as np
import cv2
from lwganrt.models.holoportator_rt import HoloportatorRT
from lwganrt.options.test_options import TestOptions
from lwganrt.models.holoportator_rt import prepare_input as prepare_lwgan_input

class HoloLwganRT:
    def __init__(self, args, warmup=True):
        assert torch.cuda.is_available()
        self.device = torch.device('cuda:' + str(args.gpu_ids))
        self.holoport_model = HoloportatorRT(args, device=self.device)
        self.personalized = False
        self.desc_frame = None
        self.desc_smpl = None
        self.mode = 'view'

    def load_descriptor(self,img_path, smpl_path):
        self.desc_frame = torch.load(img_path).to(self.device)
        self.desc_smpl = torch.load(smpl_path).to(self.device)
        self.mode = 'predefined'

    def inference(self, frame, smpl, view):
        frame = frame.to(self.device)
        smpl = smpl.to(self.device)

        # Personalize model:
        if self.personalized == False:
            if self.desc_frame is not None:
                frame = self.desc_frame
                smpl = self.desc_smpl
                self.personalized = True
            self.holoport_model.personalize(frame, smpl)

        # Inference:
        if self.mode == 'view':
            preds = self.holoport_model.view(view['R'], view['t'])
        else:
            preds = self.holoport_model.inference(smpl)

        return preds[0]

    def inference_batch(self, frames, smpls):
        assert len(frames) == len(smpls)
        frame_batch = torch.cat(frames, dim=0).to(self.device)
        smpl_batch = torch.cat(smpls, dim=0).to(self.device)

        # Personalize model:
        if self.personalized == False:
            if self.desc_frame is not None:
                frame_batch = torch.cat([self.desc_frame], dim=0).to(self.device)
                smpl_batch = torch.cat([self.desc_smpl], dim=0).to(self.device)
                self.personalized = True
            self.holoport_model.personalize(frame_batch, smpl_batch)

        # Inference:
        if self.mode == 'view':
            raise NotImplementedError
        else:
            preds = self.holoport_model.inference(smpl_batch)

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
    args = TestOptions().parse(lwgan_conf, set_cuda_env=False, verbose=False)

    # Init LWGAN-RT model:
    print('Initializing LWGAN-RT...')
    lwgan = HoloLwganRT(args)

    return lwgan, args


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
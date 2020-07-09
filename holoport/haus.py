import argparse
import torch
import numpy as np
import cv2
from holoport.utils import norm_face


class HoloActionUnitsRT():
    def __init__(self, args, warmup=True):
        # Set device:
        self.device = torch.device('cuda:' + str(args.au_gpu_id) if args.au_gpu_id is not None else 'cpu')

        # Init model:
        model = torch.jit.load(args.au_model)
        model.eval()
        self.model = model.to(self.device)

    def inference(self, frame, clip_aus=True):
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)

        # Move to device:
        frame = frame.to(self.device)

        # Forward pass:
        result = self.model(frame)
        aus = result.detach().cpu().numpy()

        if clip_aus:
            aus = np.clip(aus, a_min=0.0, a_max=1.0)

        return aus


def prepare_input(frame, landmarks, face_size=224):
    # Crop and normalize face:
    pts = [(landmarks[0], landmarks[1]), (landmarks[2], landmarks[3])]  # left and right eyes
    face_img, _ = norm_face(frame, pts, crop_type='head', max_size=face_size)

    face_img = face_img[..., ::-1]
    face_img = face_img.astype(np.float32) / 255.0
    face_img = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0)

    return face_img


def init_aus(conf):
    # Parse params:
    parser = argparse.ArgumentParser(description='ActionUnits')
    parser.add_argument('--au_model', default='', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--au_face_size', default=224, type=int, help='input face size')
    parser.add_argument('--au_make_dict', action="store_true", default=False,
                        help='output raw AUs vector or make AUs dictionary containing AU names')
    parser.add_argument('--au_gpu_id', default='0', type=str, help='CUDA device id')
    args, _ = parser.parse_known_args()

    # Set params:
    args.au_model = conf['model_path']
    args.au_face_size = conf['face_size']
    args.au_make_dict = conf['make_dict']
    args.au_gpu_id = conf['gpu_id']

    # Init vibe:
    print('Initializing ActionUnits...')
    aus_model = HoloActionUnitsRT(args)
    print('Loaded pretrained ActionUnits weights from', args.au_model)

    return aus_model, args


def make_aus_dict(raw_aus):
    names = ['brow_down_left','brow_down_right','brow_inner_up','brow_outer_up_left',
             'brow_outer_up_right','cheek_puff','cheek_squint_left','cheek_squint_right',
             'eye_blink_left','eye_blink_right','eye_look_down_left','eye_look_down_right',
             'eye_look_in_left','eye_look_in_right','eye_look_out_left','eye_look_out_right',
             'eye_look_up_left','eye_look_up_right','eye_squint_left','eye_squint_right',
             'eye_wide_left','eye_wide_right','jaw_forward','jaw_left',
             'jaw_open','jaw_right','mouth_close','mouth_dimple_left',
             'mouth_dimple_right','mouth_frown_left','mouth_frown_right','mouth_funnel',
             'mouth_left','mouth_lower_down_left','mouth_lower_down_right','mouth_press_left',
             'mouth_press_right','mouth_pucker','mouth_right','mouth_roll_lower',
             'mouth_roll_upper','mouth_shrug_lower','mouth_shrug_upper','mouth_smile_left',
             'mouth_smile_right','mouth_stretch_left','mouth_stretch_right','mouth_upper_up_left',
             'mouth_upper_up_right','nose_sneer_left','nose_sneer_right','tongue_out']
    assert raw_aus.shape[0] == len(names)
    aus_dict = {}
    for i in range(raw_aus.shape[0]):
        aus_dict[names[i]] = raw_aus[i]

    return aus_dict


def vizualize_aus(img, aus, dets):
    for au, det in zip(aus, dets):
        if isinstance(au, dict):
            eye_blink_left = au['eye_blink_right']**0.5
            jaw_open = au['jaw_open']
            eye_blink_right = au['eye_blink_left']**0.5
        else:
            eye_blink_left = au[8]
            eye_blink_right = au[9]
            jaw_open = au[24]

        box = det['box']
        height = box[3] - box[1]
        bin_h = int(height*0.3)
        bin_w = int(bin_h*0.25)
        y = box[1] + int(height*0.5)
        x = box[2] + 3

        # eye_blink_left
        cv2.rectangle(img, (x, y - bin_h), (x + bin_w, y), (128, 128, 128), -1)
        bin_y = int(eye_blink_left*bin_h)
        cv2.rectangle(img, (x, y - bin_y), (x + bin_w, y), (0, 255, 0), -1)

        # jaw_open
        x = x + bin_w + int(bin_w*0.5)
        cv2.rectangle(img, (x, y - bin_h), (x + bin_w, y), (128, 128, 128), -1)
        bin_y = int(jaw_open * bin_h)
        cv2.rectangle(img, (x, y - bin_y), (x + bin_w, y), (0, 0, 255), -1)

        # eye_blink_right
        x = x + bin_w + int(bin_w * 0.5)
        cv2.rectangle(img, (x, y - bin_h), (x + bin_w, y), (128, 128, 128), -1)
        bin_y = int(eye_blink_right * bin_h)
        cv2.rectangle(img, (x, y - bin_y), (x + bin_w, y), (255, 0, 0), -1)



    return img


if __name__ == '__main__':
    from holoport.conf.conf_parser import parse_conf
    import time
    import cv2
    import pickle

    def load_pickle_file(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data


    # Init AUs model:
    conf = parse_conf('./holoport/conf/local/aus_resnet50.yaml')
    # conf = parse_conf('./holoport/conf/local/aus_resnet34.yaml')
    aus_model, args = init_aus(conf['aus'])

    # Load an image:
    img_path = conf['input']['warmup_img']
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Load landmarks for the image:
    faces = load_pickle_file(conf['input']['warmup_face'])
    landmarks = faces[0]['landms']
    face_img = prepare_input(img_raw, landmarks, args.au_face_size)

    # 20 identical runs to test:
    aus_raw = None
    for i in range(20):
        # Inference:
        tic = time.time()
        aus_raw = aus_model.inference(face_img)
        tac = time.time()

        print('Done! Total AUs: {}, elapsed: {:.4f}'.format(aus_raw.shape, tac - tic))

    aus_dict = make_aus_dict(aus_raw[0])

    # Show the predicted AUs:
    print ('===Action Units:')
    for k,v in aus_dict.items():
        print ('{}: {:4f}'.format(k,v))
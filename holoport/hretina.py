import argparse
import torch
import numpy as np
import cv2
from retinart.data import cfg_mnet, cfg_re50
from retinart.models.retinaface import RetinaFace
from retinart.layers.functions.prior_box import PriorBox
from retinart.utils.box_utils import decode, decode_landm
from retinart.utils.nms.py_cpu_nms import py_cpu_nms


def load_model(model, pretrained_path, device):
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
        sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


class HoloRetinaRT():
    def __init__(self, args, warmup=True):
        # Set device:
        self.device = torch.device('cuda:' + str(args.rf_gpu_id) if not args.rf_cpu else 'cpu')

        # Init model:
        self.cfg = None
        if args.rf_network == "mobile0.25":
            self.cfg = cfg_mnet
            self.cfg['pretrain_mobilenet'] = args.rf_pretrain_mobilenet
        elif args.rf_network == "resnet50":
            self.cfg = cfg_re50
        net = RetinaFace(cfg=self.cfg, phase='test')
        net = load_model(net, args.rf_trained_model, self.device)
        net.eval()
        if args.cudnn_benchmark:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            print ('cudnn benchmark is enabled!')
        self.net = net.to(self.device)

        # Scale:
        self.resize = 1
        img_height, img_width = args.rf_img_hight, args.rf_img_width
        scale_list = [img_width, img_height, img_width, img_height]
        scale = torch.Tensor(scale_list)
        self.scale = scale.to(self.device)
        scale1_list = [img_width, img_height, img_width, img_height,
                       img_width, img_height, img_width, img_height,
                       img_width, img_height]
        scale1 = torch.Tensor(scale1_list)
        self.scale1 = scale1.to(self.device)

        # PriorBox:
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

        if warmup:
            dummy = torch.zeros((1, 3, args.rf_img_hight, args.rf_img_width))
            self.inference(dummy)


    def inference(self, frame):
        with torch.no_grad():
            if frame.ndimension() == 3:
                frame = frame.unsqueeze(0)

            # Move to device:
            frame = frame.to(self.device)

            # Forward pass:
            loc, conf, landms = self.net(frame)

            # Decode the predictions:
            boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
            boxes = boxes * self.scale / self.resize
            boxes = boxes.cpu().numpy()

            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            landms = landms * self.scale1 / self.resize
            landms = landms.cpu().numpy()

            return boxes, landms, scores


def prepare_input(img_raw, target_width=None, target_height=None):
    if target_width is not None and target_height is not None and \
                    target_width != img_raw.shape[1] or target_height != img_raw.shape[0]:
        interp = cv2.INTER_AREA if target_width < img_raw.shape[1] else cv2.INTER_CUBIC
        img_raw = cv2.resize(img_raw, (target_width,target_height), interpolation=interp)

    img = np.float32(img_raw)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    return img


def filter_detections(boxes, landms, scores, args):
    # ignore low scores
    inds = np.where(scores > args.rf_confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.rf_top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.rf_nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.rf_keep_top_k, :]
    landms = landms[:args.rf_keep_top_k, :]

    results = []
    for b, l in zip(dets,landms):
        prob = b[4]
        if prob < args.rf_det_threshold:
            continue
        box = list(map(int, b[:4]))
        landmarks = list(map(int, l[:10]))
        det = {'box': box,
               'prob': prob,
               'landms': landmarks}
        results.append(det)

    #  Choose the preson with the largest bbox area:
    if args.rf_main_face and len(results):
        areas = [(det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1]) for det in results]
        max_id = max(enumerate(areas), key=lambda x: x[1])[0]
        results = [results[max_id]]

    return results


def init_retina(conf):
    # Parse params:
    parser = argparse.ArgumentParser(description='Retina')
    parser.add_argument('-m', '--rf_trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--rf_pretrain_mobilenet', default='./weights/mobilenetV1X0.25_pretrain.tar',
                        type=str, help='Pretrained mobilenet file path to open')
    parser.add_argument('--rf_network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--rf_cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--rf_confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--rf_top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--rf_nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--rf_keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--rf_det_threshold', default=0.6, type=float, help='detection threshold')
    parser.add_argument('--rf_img_width', default=1920, type=int, help='img_width')
    parser.add_argument('--rf_img_hight', default=1080, type=int, help='img_hight')
    parser.add_argument('--rf_gpu_id', default='0', type=str, help='CUDA device id')
    parser.add_argument('--cudnn_benchmark', action="store_true", default=False,
                        help='Enable cudnn benchmark. It has impact when using mobilenet')
    parser.add_argument('--rf_main_face', action="store_true", default=False,
                        help='Retina returns only the largest face in the frame')
    args, _ = parser.parse_known_args()

    # Set params:
    args.rf_trained_model = conf['model_path']
    args.rf_pretrain_mobilenet = conf['pretrain_mobilenet']
    args.rf_network = conf['network']
    args.rf_det_threshold = conf['threshold']
    args.rf_img_width = conf['img_width']
    args.rf_img_hight = conf['img_hight']
    args.rf_gpu_id = conf['gpu_id']
    args.cudnn_benchmark = conf['cudnn_benchmark']
    args.rf_main_face = conf['main_face']

    # Init vibe:
    print('Initializing RetinaFace...')
    retina = HoloRetinaRT(args)
    print('Loaded pretrained RetinaFace weights from', args.rf_trained_model)

    return retina, args


def vizualize_detections(img, detections):
    for det in detections:
        box = det['box']
        prob = det['prob']
        landms = det['landms']

        text = "{:.4f}".format(prob)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cx = box[0]
        cy = box[1] + 12
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landmarks:
        cv2.circle(img, (landms[0], landms[1]), 1, (0, 0, 255), 4)
        cv2.circle(img, (landms[2], landms[3]), 1, (0, 255, 255), 4)
        cv2.circle(img, (landms[4], landms[5]), 1, (255, 0, 255), 4)
        cv2.circle(img, (landms[6], landms[7]), 1, (0, 255, 0), 4)
        cv2.circle(img, (landms[8], landms[9]), 1, (255, 0, 0), 4)

    return img


if __name__ == '__main__':
    from holoport.conf.conf_parser import parse_conf
    import time
    import os
    import pickle

    def write_pickle_file(pkl_path, data_dict):
        with open(pkl_path, 'wb') as fp:
            pickle.dump(data_dict, fp, protocol=2)


    # Init Retina:
    conf = parse_conf('./holoport/conf/local/retina-resnet50.yaml')
    # conf = parse_conf('./holoport/conf/local/retina-mobilenet.yaml')
    retina, args = init_retina(conf['retina'])

    # Load an image:
    img_path = conf['input']['warmup_img']
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = prepare_input(img_raw, conf['retina']['img_width'], conf['retina']['img_hight'])

    # 20 identical runs to test:
    result_img, detections = None, None
    for i in range(20):
        # Inference:
        tic = time.time()
        boxes, landms, scores = retina.inference(img)

        # Filter detections on CPU:
        detections = filter_detections(boxes, landms, scores, args)
        tac = time.time()

        # Visualize the detections:
        result_img = vizualize_detections(img_raw, detections)

        print('Done! Total detections: {}, elapsed: {:.4f}'.format(len(detections), tac - tic))

    # Save the result image:
    if not os.path.exists(conf['output']['result_dir']):
        os.makedirs(conf['output']['result_dir'])
    dst_path = os.path.join(conf['output']['result_dir'], 'test.jpeg')
    cv2.imwrite(dst_path, result_img)
    dst_path = os.path.join(conf['output']['result_dir'], 'test_faces.pkl')
    write_pickle_file(dst_path, detections)

    print ('The results have been saved to', dst_path)
import torch
import argparse
import numpy as np
from yolort.models import Darknet
from yolort.utils.utils import non_max_suppression, scale_coords, load_classes
from yolort.utils.datasets import letterbox


class HoloYoloRT():
    def __init__(self, args, warmup=True):
        self.device = torch.device('cuda:' + str(args.yolo_gpu_id))
        self.half = args.yolo_half
        self.img_size = args.yolo_img_size
        self.conf_thres = args.yolo_conf_thres
        self.iou_thres = args.yolo_iou_thres
        self.classes = args.yolo_classes
        self.agnostic_nms = args.yolo_agnostic_nms

        # Init YOLO-RT:
        self.yolo_model = Darknet(args.yolo_cfg, self.img_size)
        self.yolo_model.load_state_dict(torch.load(args.yolo_weights, map_location=self.device)['model'])
        self.yolo_model.to(self.device).eval()

        # Half precision
        if self.half:
            self.yolo_model.half()

        # Get class names
        self.names = load_classes(args.yolo_names)

        # Warm up:
        if warmup:
            with torch.no_grad():
                dummy = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
                _ = self.yolo_model(dummy.half() if self.half else dummy.float())


    def inference(self, frame):
        with torch.no_grad():
            if frame.ndimension() == 3:
                frame = frame.unsqueeze(0)

            # To GPU:
            frame = frame.to(self.device)

            if self.half and frame.dtype != torch.half:
                frame = frame.half()

            # Inference
            pred = self.yolo_model(frame, augment=False)[0]

            if self.half:
                pred = pred.float()

            # Apply NMS:
            nms_pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                           multi_label=False, classes=self.classes, agnostic=self.agnostic_nms)

            # To CPU:
            pred = []
            for p in nms_pred:
                if p is not None:
                    pred.append(p.cpu())
                else:
                    pred.append(None)

            return pred


def prepare_yolo_input(img, img_size):
    # Padded resize:
    img = letterbox(img, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # To tensor:
    img = torch.from_numpy(img)
    img = img.float()

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def convert_yolo_output_to_bboxes(pred, actual_size, origin_size, target_class=0):
    assert len(pred) == len(actual_size) and len(pred) == len(origin_size)
    bboxes = []

    for det, size1, size0 in zip(pred, actual_size, origin_size):
        bbox = None

        if det is not None and len(det):
            # Rescale boxes from img_size to the origin size
            det[:, :4] = scale_coords(size1, det[:, :4], size0).round()

            # Filter out all not humans:
            persons = []
            for i in range(det.shape[0]):
                if int(det[i,5]) == target_class:
                    persons.append(det[i])

            #  Choose the preson with the largest bbox area:
            if len(persons):
                areas = [(p[2]-p[0])*(p[3]-p[1]) for p in persons]
                max_id = max(enumerate(areas), key=lambda x: x[1])[0]
                bbox = persons[max_id][:4]

        bboxes.append(bbox)

    return bboxes


def init_yolo(yolo_conf):
    # Init params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_cfg', type=str, default='yolort/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--yolo_names', type=str, default='yolort/data/coco.names', help='*.names path')
    parser.add_argument('--yolo_weights', type=str, default='yolort/weights/yolov3-spp-ultralytics.pt',
                        help='yolo_v3 weights path')
    parser.add_argument('--yolo_img_size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--yolo_conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--yolo_iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--yolo_half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--yolo_gpu_id', type=str, default='0', help='device id')
    parser.add_argument('--yolo_classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--yolo_agnostic_nms', action='store_true', help='class-agnostic NMS')
    args, _ = parser.parse_known_args()

    # Set params:
    args.yolo_cfg = yolo_conf['cfg_path']
    args.yolo_names = yolo_conf['names_path']
    args.yolo_weights = yolo_conf['weights_path']
    args.yolo_gpu_id = yolo_conf['gpu_id']
    args.yolo_img_size = yolo_conf['img_size']
    if 'half' in yolo_conf:
        args.yolo_half = yolo_conf['half']

    # Init vibe:
    print('Initializing YOLOv3...')
    yolo = HoloYoloRT(args)
    print('Loaded pretrained YOLOv3 weights from', args.yolo_weights)

    return yolo, args


def pre_yolo(data, args):
    # Prepare YOLO input:
    data['yolo_input'] = prepare_yolo_input(data['frame'].copy(), args.yolo_img_size)

    return data


def post_yolo(data):
    # Prepare YOLO output:
    actual_size = [data['yolo_input'].shape[2:]]
    origin_size = [data['frame'].shape]
    bboxes = convert_yolo_output_to_bboxes(data['yolo_output'], actual_size, origin_size)

    # Covert predicted bbox:
    if len(bboxes) and bboxes[0] is not None:
        # (x1,y1,x2,y2) -> (x1,y1,w,h)
        bbox = bboxes[0].clone().numpy()
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        # (x1,y1,w,h) -> (x,y,s,s), s=max(w,h), x=x1-dx, y=y1-dy
        side = max(bbox[2], bbox[3])
        dx = (side - bbox[2]) * 0.5
        dy = (side - bbox[3]) * 0.5
        bbox[0] = bbox[0] - dx
        bbox[1] = bbox[1] - dy
        bbox[2] = side
        bbox[3] = side
        data['yolo_bbox'] = np.expand_dims(bbox, axis=0)

        # (x,y,w,h) -> (cx,cy,w,h)
        yolo_cbbox = np.copy(data['yolo_bbox'])
        yolo_cbbox[:,0] = yolo_cbbox[:,0] + yolo_cbbox[:,2] * 0.5
        yolo_cbbox[:,1] = yolo_cbbox[:,1] + yolo_cbbox[:,3] * 0.5
        data['yolo_cbbox'] = yolo_cbbox
    else:
        data['yolo_bbox'] = None
        data['yolo_cbbox'] = None

    return data
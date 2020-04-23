import torch
import argparse
import numpy as np
from yolort.models import Darknet
from yolort.utils.utils import non_max_suppression, scale_coords, load_classes
from yolort.utils.datasets import letterbox


class HoloYoloRT():
    def __init__(self, args):
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
        with torch.no_grad():
            _ = self.yolo_model(torch.zeros((1, 3, self.img_size, self.img_size), device=self.device))


    def inference(self, frame):
        with torch.no_grad():
            # Prepare input:
            if frame.ndimension() == 3:
                frame = frame.unsqueeze(0)
            frame = frame.to(self.device)

            # Inference
            pred = self.yolo_model(frame, augment=False)[0]

            if self.half:
                pred = pred.float()

            # Apply NMS:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       multi_label=False, classes=self.classes, agnostic=self.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to the origin size
                    det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame.shape).round()

            return {
                'bboxes': pred
            }


def prepare_yolo_input(img, img_size, half=False):
    # Padded resize:
    img = letterbox(img, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # To tensor:
    img = torch.from_numpy(img)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


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

    # Init vibe:
    print('Initializing YOLOv3...')
    yolo = HoloYoloRT(args)
    print('Loaded pretrained YOLOv3 weights from', args.yolo_weights)

    return yolo, args
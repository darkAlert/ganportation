import os
import cv2
import time
import sys
import numpy as np
from vibert.lib.data_utils.img_utils import get_single_image_crop_demo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hvibe import init_vibe, convert_cam
from conf.conf_parser import parse_conf


class ExampleModel(object):
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf='conf/azure/vibe_conf_azure.yaml'):
        # Load config:
        conf = parse_conf(path_to_conf)
        print('Config has been loaded from', path_to_conf)

        # Init VIBE-RT model:
        self.vibe, self.args = init_vibe(conf['vibe'])

        self.connector = connector
        self.finished = time.time()
        self.time = None

        # Dummy data only for tests:
        self.dummy_yolo_cbboxes = np.array([[300.0, 300.0, 300.0, 300.0]])
        self.dummy_scene_cbbox = np.array([[350.0, 350.0, 350.0, 350.0]])


    def run(self):
        for frame in self.connector.frames():
            # Preprocess input data:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            norm_img, _, _ = get_single_image_crop_demo(img, self.dummy_yolo_cbboxes,
                                                        kp_2d=None,
                                                        scale=self.args.scale,
                                                        crop_size=self.args.crop_size)
            data = {
                'vibe_input': norm_img.unsqueeze(0),
                'yolo_cbbox': self.dummy_yolo_cbboxes,
                'scene_cbbox': self.dummy_scene_cbbox
            }

            # Inference:
            output = self.vibe.inference(data['vibe_input'])

            avatar_cam = convert_cam(cam=output['pred_cam'].numpy(),
                                     bbox1=data['yolo_cbbox'],
                                     bbox2=data['scene_cbbox'],
                                     truncated=True)
            data['smpl'] = {
                'pred_cam': output['pred_cam'].numpy(),
                'pose': output['pose'].numpy(),
                'betas': output['betas'].numpy(),
                'rotmat': output['rotmat'].numpy(),
                'avatar_cam': avatar_cam,
            }


            # Send the result:
            waited = time.time() - self.finished

            spf = 100
            if self.time is None:
                self.time = time.time()
            else:
                now = time.time()
                spf, self.time = now - self.time, now

            fps = 1/spf
            self.connector.send_data(dict(fps=fps, point='{}'.format(img[0,0])))

            free_time = waited/(time.time() - self.finished)
            text = '{:.1f}'.format(free_time)
            cv2.putText(img, text, (5, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            text = '{:.3f}/{:.3f}/{:.3f}'.format(data['avatar_cam'][0], data['avatar_cam'][1], data['avatar_cam'][2])
            cv2.putText(img, text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            self.connector.send_frame(img)
            self.finished = time.time()

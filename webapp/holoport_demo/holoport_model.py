import os
import numpy as np
import cv2
import threading
from queue import Queue
from holoport.workers import *
from holoport.conf.conf_parser import parse_conf
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.hvibe import init_vibe
from holoport.hyolo import init_yolo


def send_worker(break_event, avatar_q, send_data, send_frame, timeout=0.005):
    print('send_worker has been run...')

    # Make not_found frame:
    not_found_frame = np.zeros((256,256,3),dtype=np.uint8)
    text = 'Person not found!'
    pos = (10, 120)
    cv2.putText(not_found_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)

    start = time.time()

    while not break_event.is_set():
        try:
            data = avatar_q.get(timeout=timeout)
            avatar_q.task_done()
        except Empty:
            continue

        # Measure FPS:
        stop = time.time()
        elapsed = stop - start
        start = stop
        fps = 1000 / elapsed

        if 'not_found' in data:
            # Send not_found frame:
            frame = not_found_frame.copy()
            text = '{:.1f}'.format(fps)
            cv2.putText(frame, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            send_data(dict(fps=fps, not_found=True))
            send_frame(frame)
        else:
            # Send the avatar:
            text = '{:.1f}'.format(fps)
            cv2.putText(data['avatar'], text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            send_data(dict(fps=fps))
            send_frame(data['avatar'])

    print('send_worker has been terminated.')

    return True


class HoloportModel(object):
    LABEL = ['holoport_realtime']
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf='holoport_conf_azure.yaml'):
        self.connector = connector

        # Load config:
        path_to_conf = os.path.join(os.path.dirname(__file__), path_to_conf)
        conf = parse_conf(path_to_conf)
        self.connector.logger.info('Config has been loaded from {}'.format(path_to_conf))

        # Init models:
        self.yolo, self.yolo_args = init_yolo(conf['yolo'])
        self.vibe, self.vibe_args = init_vibe(conf['vibe'])
        self.lwgan, self.lwgan_args = init_lwgan(conf['lwgan'])

        # Warmup:
        if 'warmup_img' in conf['input']:
            img = cv2.imread(conf['input']['warmup_img'], 1)
            warmup_holoport_pipeline(img, self.yolo, self.yolo_args, self.vibe,
                                     self.vibe_args, self.lwgan, self.lwgan_args)

        # Avatar view params:
        steps = conf['input']['steps']
        view = parse_view_params(conf['input']['view'])

        # Dummy scene params:
        t = conf['input']['scene_bbox'].split(',')
        assert len(t) == 4
        dummy_scene_bbox = np.array([[int(t[0]), int(t[1]), int(t[2]), int(t[3])]], dtype=np.int64)
        dummy_scene_cbbox = dummy_scene_bbox.copy()
        dummy_scene_cbbox[:, 0] = dummy_scene_bbox[:, 0] + dummy_scene_bbox[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        dummy_scene_cbbox[:, 1] = dummy_scene_bbox[:, 1] + dummy_scene_bbox[:, 3] * 0.5

        # Set auxiliary params:
        self.aux_params = {}
        self.aux_params['dummy_scene_bbox'] = dummy_scene_bbox
        self.aux_params['dummy_scene_cbbox'] = dummy_scene_cbbox
        self.aux_params['steps'] = steps
        self.aux_params['view'] = view

        # Make queues, events and threads:
        self.break_event = threading.Event()
        self.frame_q = Queue(maxsize=10000)
        self.yolo_input_q = Queue(maxsize=1000)
        self.yolo_output_q = Queue(maxsize=1000)
        self.vibe_input_q = Queue(maxsize=1000)
        self.vibe_output_q = Queue(maxsize=1000)
        self.lwgan_input_q = Queue(maxsize=1000)
        self.lwgan_output_q = Queue(maxsize=1000)
        self.avatar_q = Queue(maxsize=10000)
        self.workers = []

        # Make workers:
        worker_args = (self.yolo_args, self.break_event, self.frame_q, self.yolo_input_q, self.aux_params)
        self.workers.append(threading.Thread(target=pre_yolo_worker, args=worker_args))
        worker_args = (self.vibe_args, self.break_event, self.yolo_output_q, self.vibe_input_q)
        self.workers.append(threading.Thread(target=pre_vibe_worker, args=worker_args))
        worker_args = (self.lwgan_args, self.break_event, self.vibe_output_q, self.lwgan_input_q)
        self.workers.append(threading.Thread(target=pre_lwgan_worker, args=worker_args))
        worker_args = (self.break_event, self.lwgan_output_q, self.avatar_q)
        self.workers.append(threading.Thread(target=postprocess_worker, args=worker_args))
        worker_args = (self.yolo, self.vibe, self.break_event, self.yolo_input_q,
                       self.yolo_output_q, self.vibe_input_q, self.vibe_output_q)
        self.workers.append(threading.Thread(target=yolo_vibe_inference_worker, args=worker_args))
        worker_args = (self.lwgan, self.break_event, self.lwgan_input_q, self.lwgan_output_q, 0.005, True)
        self.workers.append(threading.Thread(target=lwgan_inference_worker, args=worker_args))
        worker_args = (self.break_event, self.avatar_q, self.connector.send_data, self.connector.send_frame)
        self.workers.append(threading.Thread(target=send_worker, args=worker_args))


    def run(self):
        self.connector.logger.info('Running holoport...')

        test_frame = np.zeros((256, 256, 3), dtype=np.uint8)

        # Run workers:
        for w in self.workers:
            w.start()

        # Cleanup frames queue:
        _ = self.connector.recv_all_frames()

        for frame in self.connector.frames():
            if frame is not None:
                # Put frame in the queue and continue:
                # self.frame_q.put({'frame': frame})
                self.frame_q.put({'frame': test_frame.copy()})

            else:
                print ('break')
                # Or terminate the process and wait for the workers:
                self.break_event.set()
                for w in self.workers:
                    w.join()
                self.connector.logger.info('Holoport model has been stopped!')
                return True
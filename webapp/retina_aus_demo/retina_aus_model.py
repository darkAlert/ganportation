import sys
import os
import cv2
import threading
from queue import Queue
from holoport.workers import *
from holoport.conf.conf_parser import parse_conf
from holoport.hretina import init_retina
from holoport.haus import init_aus
from holoport.stream import LiveStream, VideoStream


def renderer_worker(renderer_conf, break_event, input_q, output_q, timeout=0.005):
    print('renderer_worker has been run...')

    # Init renderer (need toinitialize in this thread)
    # if renderer_conf['type'] == 'TrimeshSmplRenderer':
    #     renderer = TrimeshSmplRenderer(renderer_conf)
    #     print ('TrimeshSmplRenderer is used.')
    if renderer_conf['type'] == 'NeuralSmplRenderer':
        renderer = NeuralSmplRenderer(renderer_conf)
        print('NeuralSmplRenderer is used.')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        data['avatar'] = data['frame'].copy()
        scene_bbox = data['scene_bbox'][0]

        # Check if the predicted yolo box fits into the scene box:
        if 'not_found' not in data and \
                not box_fits_into_scene(data['yolo_bbox'][0], scene_bbox):
            data['not_found'] = True
            # Draw the predicted yolo bbox:
            bbox = data['yolo_bbox'][0]
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(data['avatar'], pt1, pt2, color=(0, 255, 255), thickness=1)

        if 'not_found' in data:
            # Draw scene rectancgle:
            pt1 = (scene_bbox[0],scene_bbox[1])
            pt2 = (scene_bbox[0]+scene_bbox[2], scene_bbox[1]+scene_bbox[3])
            cv2.rectangle(data['avatar'], pt1, pt2, (0,0,255), thickness=2)
            output_q.put(data)
            continue

        # Prepare VIBE output:
        data = post_vibe(data)

        # Render SMPL:
        if not 'verts' in data['vibe_output']:
            raise NotImplementedError
        data['rendered_smpl'] = renderer.render(data['vibe_output']['verts'], data['scene_cam'],
                                                (scene_bbox[2],scene_bbox[3]))
        x1, x2 = scene_bbox[0], scene_bbox[0] + scene_bbox[2]
        y1, y2 = scene_bbox[1], scene_bbox[1] + scene_bbox[3]
        avatar_crop = cv2.addWeighted(data['avatar'][y1:y2, x1:x2, :], 0.1, data['rendered_smpl'], 0.9, 0)
        data['avatar'][y1:y2, x1:x2, :] = avatar_crop

        # Draw scene rectancgle:
        bbox = data['scene_bbox'][0]
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv2.rectangle(data['avatar'], pt1, pt2, (0, 255, 0), thickness=2)

        output_q.put(data)

    print('renderer_worker has been terminated.')

    return True


def send_worker(break_event, avatar_q, send_data, send_frame, timeout=0.005):
    print('send_worker has been run...')

    mean_latency = None
    mean_fps = None
    start = time.time()
    prev_frame_start = time.time()
    mean_wait = None

    while not break_event.is_set():
        try:
            data = avatar_q.get(timeout=timeout)
            avatar_q.task_done()
        except Empty:
            continue

        # Measure FPS and latency:
        stop = time.time()
        elapsed = stop - start
        start = stop

        fps = 1 / elapsed
        if mean_fps is None:
            mean_fps = fps
        mean_fps = mean_fps*0.9 + fps*0.1
        latency = stop - data['start']
        if mean_latency is None:
            mean_latency = latency
        mean_latency = mean_latency * 0.9 + latency * 0.1
        wait =  data['start'] - prev_frame_start
        if mean_wait is None:
            mean_wait = wait
        mean_wait = mean_wait * 0.9 + wait * 0.1
        prev_frame_start = data['start']

        # Draw:
        text = 'fps:{:.1f}'.format(mean_fps)
        cv2.putText(data['avatar'], text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        text = 'lat:{:.1f}'.format(mean_latency * 1000)
        cv2.putText(data['avatar'], text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        text = 'wait:{:.1f}'.format(mean_wait * 1000)
        cv2.putText(data['avatar'], text, (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Send data:
        send_data(dict(fps=fps))
        send_frame(data['avatar'])

    print('send_worker has been terminated.')

    return True


def generate_aux_params(conf):
    # Avatar view params:
    steps = conf['steps']
    view = parse_view_params(conf['view'])

    # Dummy scene params:
    t = conf['scene_bbox'].split(',')
    assert len(t) == 4
    scene_bbox = np.array([[int(t[0]), int(t[1]), int(t[2]), int(t[3])]], dtype=np.int64)
    scene_cbbox = scene_bbox.copy()
    scene_cbbox[:, 0] = scene_bbox[:, 0] + scene_bbox[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
    scene_cbbox[:, 1] = scene_bbox[:, 1] + scene_bbox[:, 3] * 0.5

    # Set auxiliary params:
    aux_params = {}
    aux_params['scene_bbox'] = scene_bbox
    aux_params['scene_cbbox'] = scene_cbbox
    aux_params['steps'] = steps
    aux_params['view'] = view

    return aux_params


class RetinaFaceAUsModel(object):
    LABEL = ['retinaface+aus_live']
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf='retinaface-aus_live.yaml'):
        self.connector = connector
        self.connector.enable_frame_throw()
        self.name = 'RetinaFaceAUsModel'

        # Load config:
        path_to_conf = os.path.join(os.path.dirname(__file__), path_to_conf)
        conf = parse_conf(path_to_conf)
        self.connector.logger.info('Config has been loaded from {}'.format(path_to_conf))

        # Init model:
        self.retina, self.yolo_args = init_yolo(conf['yolo'])
        self.aus, self.vibe_args = init_vibe(conf['vibe'])

        # Warmup:
        if 'warmup_img' in conf['input']:
            img = cv2.imread(conf['input']['warmup_img'], 1)
            warmup_holoport_pipeline(img, self.yolo, self.yolo_args, self.vibe, self.vibe_args)

        # Auxiliary params:
        self.aux_params = generate_aux_params(conf['input'])

        # Make queues, events and threads:
        self.break_event = threading.Event()
        self.frame_q = Queue(maxsize=10000)
        self.yolo_input_q = Queue(maxsize=1000)
        self.yolo_output_q = Queue(maxsize=1000)
        self.vibe_input_q = Queue(maxsize=1000)
        self.vibe_output_q = Queue(maxsize=1000)
        self.avatar_q = Queue(maxsize=10000)
        self.workers = []

        # Make workers:
        worker_args = (self.yolo_args, self.break_event, self.frame_q, self.yolo_input_q, self.aux_params)
        self.workers.append(threading.Thread(target=pre_yolo_worker, args=worker_args))
        worker_args = (self.vibe_args, self.break_event, self.yolo_output_q, self.vibe_input_q)
        self.workers.append(threading.Thread(target=pre_vibe_worker, args=worker_args))
        worker_args = (self.yolo, self.vibe, self.break_event, self.yolo_input_q,
                       self.yolo_output_q, self.vibe_input_q, self.vibe_output_q)
        self.workers.append(threading.Thread(target=yolo_vibe_inference_worker, args=worker_args))
        worker_args = (conf['renderer'], self.break_event, self.vibe_output_q, self.avatar_q)
        self.workers.append(threading.Thread(target=renderer_worker, args=worker_args))
        worker_args = (self.break_event, self.avatar_q, self.connector.send_data, self.connector.send_frame)
        self.workers.append(threading.Thread(target=send_worker, args=worker_args))

    def run(self):
        self.connector.logger.info('Running {}...'.format(self.name))

        # Run workers:
        for w in self.workers:
            w.start()

        # Cleanup frames queue:
        _ = self.connector.recv_all_frames()

        for idx, frame in enumerate(self.connector.frames()):
            if idx % 2 == 0:
                continue

            if self.break_event.is_set():
                break

            if frame is not None:
                data = {'frame': frame.copy(), 'start': time.time()}
                self.frame_q.put(data, timeout=0.005)
            else:
                self.stop()

    def stop(self):
        self.connector.logger.info('Stopping {}...'.format(self.name))
        self.break_event.set()

        for w in self.workers:
            w.join()

        self.connector.logger.info('{} has been stopped!'.format(self.name))

        return True


def main(path_to_conf):
    output_dir = None#'/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/rt/yolo_vibe/live'
    live = LiveStream(output_dir)
    live.run_model(YoloVibeModel, path_to_conf=path_to_conf)

if __name__ == '__main__':
    path_to_conf = 'yolo-vibe-nr_live.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    main(path_to_conf)

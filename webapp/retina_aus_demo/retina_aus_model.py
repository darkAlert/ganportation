import sys
import os
import cv2
import threading
import time
from queue import Queue, Empty
from holoport.conf.conf_parser import parse_conf
from holoport.hretina import init_retina, filter_detections, prepare_input as prepare_retina_input
from holoport.hretina import vizualize_detections
from holoport.haus import init_aus, make_aus_dict, vizualize_aus, prepare_input as prepare_aus_input
from holoport.stream import LiveStream, VideoStream
from holoport.utils import increase_brightness


def pre_retina_worker(args, break_event, input_q, output_q, timeout=0.005):
    print('pre_retina_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Prepare RetinaFace input:
        data['retina_input'] = prepare_retina_input(data['frame'], args.rf_img_width, args.rf_img_hight)
        output_q.put(data)

    print('pre_retina_worker has been terminated.')

    return True

def retina_inference_worker(retina, lock, break_event, input_q, output_q, timeout=0.005):
    print('retina_inference_worker has been run...')

    while not break_event.is_set():
        try:
            data= input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Inference:
        lock.acquire()
        boxes, landms, scores = retina.inference(data['retina_input'])
        lock.release()
        data['raw_dets'] = (boxes, landms, scores)
        output_q.put(data)

    print('retina_inference_worker has been terminated.')

    return True

def pre_aus_worker(args_retina, args_aus, break_event, input_q, output_q, timeout=0.005):
    print('pre_aus_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Filter the RetinaFace detections:
        boxes, landms, scores = data['raw_dets']
        data['dets'] = filter_detections(boxes, landms, scores, args_retina)

        # Prepare AUs input:
        faces = []
        for det in data['dets']:
            face_img = prepare_aus_input(data['frame'], det['landms'], args_aus.au_face_size)
            faces.append(face_img)
        data['faces'] = faces
        output_q.put(data)

    print('pre_aus_worker has been terminated.')

    return True

def aus_inference_worker(aus_model, args, lock, break_event, input_q, output_q, timeout=0.005):
    print('aus_inference_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Inference:
        lock.acquire()
        aus = []
        for face_img in data['faces']:
            aus.append(aus_model.inference(face_img)[0])
        lock.release()

        if args.au_make_dict:
            aus_dict = []
            for au in aus:
                aus_dict.append(make_aus_dict(au))
            aus = aus_dict
        data['aus'] = aus
        output_q.put(data)

    print('aus_inference_worker has been terminated.')

    return True


def send_worker(break_event, input_q, send_data, send_frame, timeout=0.005):
    print('send_worker has been run...')

    mean_latency = None
    mean_fps = None
    start = time.time()
    prev_frame_start = time.time()
    mean_wait = None

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
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

        # Draw boxes and aus:
        data['result_frame'] = vizualize_detections(data['frame'], data['dets'])
        data['result_frame'] = vizualize_aus(data['result_frame'], data['aus'], data['dets'])

        # Draw FPS:
        text = 'fps:{:.1f}'.format(mean_fps)
        cv2.putText(data['result_frame'], text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        text = 'lat:{:.1f}'.format(mean_latency * 1000)
        cv2.putText(data['result_frame'], text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        text = 'wait:{:.1f}'.format(mean_wait * 1000)
        cv2.putText(data['result_frame'], text, (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Send data:
        send_data(dict(fps=fps))
        send_frame(data['result_frame'])

    print('send_worker has been terminated.')

    return True


class RetinaAUsModel(object):
    LABEL = ['retina+aus_live']
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf='retina-aus_live.yaml'):
        self.connector = connector
        self.connector.enable_frame_throw()
        self.name = 'RetinaAUsModel'

        # Load config:
        path_to_conf = os.path.join(os.path.dirname(__file__), path_to_conf)
        conf = parse_conf(path_to_conf)
        self.connector.logger.info('Config has been loaded from {}'.format(path_to_conf))

        # Init models:
        self.retina, self.retina_args = init_retina(conf['retina'])
        self.aus_model, self.aus_args = init_aus(conf['aus'])

        # Warmup:
        if 'warmup_img' in conf['input']:
            img = cv2.imread(conf['input']['warmup_img'], 1)
            # warmup_holoport_pipeline(img, self.yolo, self.yolo_args, self.vibe, self.vibe_args)

        # Make queues, events and threads:
        self.break_event = threading.Event()
        self.frame_q = Queue(maxsize=10000)
        self.retina_input_q = Queue(maxsize=1000)
        self.retina_output_q = Queue(maxsize=1000)
        self.aus_input_q = Queue(maxsize=1000)
        self.aus_output_q = Queue(maxsize=1000)
        self.workers = []
        self.lock = threading.Lock()

        # Make workers:
        worker_args = (self.retina_args, self.break_event, self.frame_q, self.retina_input_q)
        self.workers.append(threading.Thread(target=pre_retina_worker, args=worker_args))
        worker_args = (self.retina, self.lock, self.break_event, self.retina_input_q, self.retina_output_q)
        self.workers.append(threading.Thread(target=retina_inference_worker, args=worker_args))
        worker_args = (self.retina_args, self.aus_args, self.break_event, self.retina_output_q, self.aus_input_q)
        self.workers.append(threading.Thread(target=pre_aus_worker, args=worker_args))
        worker_args = (self.aus_model, self.aus_args, self.lock, self.break_event, self.aus_input_q, self.aus_output_q)
        self.workers.append(threading.Thread(target=aus_inference_worker, args=worker_args))
        worker_args = (self.break_event, self.aus_output_q, self.connector.send_data, self.connector.send_frame)
        self.workers.append(threading.Thread(target=send_worker, args=worker_args))

    def run(self):
        self.connector.logger.info('Running {}...'.format(self.name))

        # Run workers:
        for w in self.workers:
            w.start()

        # Cleanup frames queue:
        _ = self.connector.recv_all_frames()

        for idx, frame in enumerate(self.connector.frames()):
            if self.break_event.is_set():
                break

            if frame is not None:
                frame = increase_brightness(frame, 10)
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
    live.run_model(RetinaAUsModel, path_to_conf=path_to_conf)

if __name__ == '__main__':
    path_to_conf = 'retina-aus_live.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    main(path_to_conf)

import sys
import os
import cv2
from tqdm import tqdm
import threading
from queue import Queue
from holoport.workers import *
from holoport.conf.conf_parser import parse_conf
from holoport.hyolo import init_yolo
from holoport.hvibe import init_vibe
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.stream import LiveStream, VideoStream, VideoSaver
from holoport.utils import increase_brightness
from holoport.trainer import LWGANTrainer
from lwganrt.models.segmentator_rt import SegmentatorRT
import lwganrt.utils.cv_utils as cv_utils
from lwganrt.utils.util import write_pickle_file, clear_dir
from lwganrt.options.train_options import TrainOptions


def send_worker(break_event, dataset_is_ready, dataset, dataset_size, input_q, send_data, send_frame, timeout=0.005):
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

        # Draw:
        scene_bbox = data['scene_bbox'][0]
        pt1 = (scene_bbox[0], scene_bbox[1])
        pt2 = (scene_bbox[0] + scene_bbox[2], scene_bbox[1] + scene_bbox[3])
        if 'not_found' in data:
            # Draw scene area:
            cv2.rectangle(data['frame'], pt1, pt2, (0,0,255), thickness=2)

            if data['yolo_bbox'] is not None:
                # Draw the predicted yolo bbox:
                bbox = data['yolo_bbox'][0]
                pt1 = (bbox[0], bbox[1])
                pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                cv2.rectangle(data['frame'], pt1, pt2, color=(0, 255, 255), thickness=1)
        else:
            # Draw scene area:
            cv2.rectangle(data['frame'], pt1, pt2, (0, 255, 0), thickness=2)

            # Dataset queue:
            dataset.append(data)
            if len(dataset) >= dataset_size:
                dataset_is_ready.set()
                break

        # Draw info (fps and etc.):
        text = 'fps:{:.1f}'.format(mean_fps)
        cv2.putText(data['frame'], text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        text = 'lat:{:.1f}'.format(mean_latency * 1000)
        cv2.putText(data['frame'], text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        text = 'wait:{:.1f}'.format(mean_wait * 1000)
        cv2.putText(data['frame'], text, (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Send data:
        send_data(dict(fps=fps))
        send_frame(data['frame'])

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

class HoloportAdaModel(object):
    LABEL = ['holoport_adaptive_training']
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf=None):
        self.connector = connector
        self.connector.enable_frame_throw()
        self.name = 'HoloportAdaModel'

        # Load config:
        if path_to_conf is None:
                path_to_conf = 'adaptive_training.yaml'
        path_to_conf = os.path.join(os.path.dirname(__file__), path_to_conf)
        self.conf = parse_conf(path_to_conf)
        self.connector.logger.info('Config has been loaded from {}'.format(path_to_conf))

        # FPS:
        if 'target_fps' in self.conf['input']:
            self.target_fps = self.conf['input']['target_fps']
        else:
            self.target_fps = 15
        self.ms_per_frame = 1.0 / self.target_fps

        # Init model:
        self.yolo, self.yolo_args = init_yolo(self.conf['yolo'])
        self.vibe, self.vibe_args = init_vibe(self.conf['vibe'])
        # self.lwgan, self.lwgan_args = init_lwgan(conf['lwgan'])
        # self.lwgan.mode = 'predefined'
        self.pre_lwgan_args = lambda:0
        self.pre_lwgan_args.image_size = self.conf['lwgan_ada']['image_size']
        self.dataset = []

        # Warmup:
        if 'warmup_img' in self.conf['input']:
            img = cv2.imread(self.conf['input']['warmup_img'], 1)
            warmup_holoport_pipeline(img, self.yolo, self.yolo_args,
                                     self.vibe, self.vibe_args)

        # Auxiliary params:
        self.aux_params = generate_aux_params(self.conf['input'])

        # Make queues, events and threads:
        self.break_event = threading.Event()
        self.dataset_is_ready = threading.Event()
        self.frame_q = Queue(maxsize=1000)
        self.yolo_input_q = Queue(maxsize=1000)
        self.yolo_output_q = Queue(maxsize=1000)
        self.vibe_input_q = Queue(maxsize=1000)
        self.vibe_output_q = Queue(maxsize=1000)
        self.output_q = Queue(maxsize=1000)
        self.dataset_q = Queue(maxsize=1000)
        self.workers = []

        # Make workers:
        worker_args = (self.yolo_args, self.break_event, self.frame_q, self.yolo_input_q, self.aux_params)
        self.workers.append(threading.Thread(target=pre_yolo_worker, args=worker_args))
        worker_args = (self.vibe_args, self.break_event, self.yolo_output_q, self.vibe_input_q, 0.005, True)
        self.workers.append(threading.Thread(target=pre_vibe_worker, args=worker_args))
        worker_args = (self.pre_lwgan_args, self.break_event, self.vibe_output_q, self.output_q)
        self.workers.append(threading.Thread(target=pre_lwgan_worker, args=worker_args))
        worker_args = (self.yolo, self.vibe, self.break_event, self.yolo_input_q,
                       self.yolo_output_q, self.vibe_input_q, self.vibe_output_q)
        self.workers.append(threading.Thread(target=yolo_vibe_inference_worker, args=worker_args))
        worker_args = (self.break_event, self.dataset_is_ready, self.dataset, self.conf['lwgan_ada']['dataset_size'],
                       self.output_q, self.connector.send_data, self.connector.send_frame)
        self.workers.append(threading.Thread(target=send_worker, args=worker_args))

        # worker_args = (self.segm, self.break_event, self.segm_input_q, self.segm_output_q)
        # self.workers.append(threading.Thread(target=segm_inference_worker, args=worker_args))
        # worker_args = (conf['lwgan_ada'], self.break_event, self.dataset_is_ready, self.segm_output_q, self.output_q)
        # self.workers.append(threading.Thread(target=dataset_worker, args=worker_args))
        # worker_args = (self.lwgan, self.break_event, self.lwgan_input_q, self.lwgan_output_q, 0.005, True)
        # self.workers.append(threading.Thread(target=lwgan_inference_worker, args=worker_args))


    def run(self):
        self.connector.logger.info('Running {}...'.format(self.name))
        self.dataset_is_ready.set()

        # Run workers:
        for w in self.workers:
            w.start()

        # Cleanup frames queue:
        _ = self.connector.recv_all_frames()
        start = time.time()

        for idx, frame in enumerate(self.connector.frames()):
            # Rectify FPS:
            elapsed = time.time() - start
            if elapsed < self.ms_per_frame:
                continue
            _ = self.connector.recv_all_frames()
            start = time.time()

            if self.break_event.is_set():
                break

            if self.dataset_is_ready.is_set():
                self.stop()
                break

            if frame is not None:
                frame = increase_brightness(frame, 10)
                data = {'frame': frame.copy(), 'start': time.time()}
                self.frame_q.put(data, timeout=0.005)
            else:
                self.stop()

        # Run adaptive training:
        if self.dataset_is_ready.is_set():
            self.connector.logger.info('Starting adaprive training process...')
            self.run_ada()


    def stop(self):
        self.connector.logger.info('Stopping {}...'.format(self.name))
        self.break_event.set()

        for w in self.workers:
            w.join()

        self.connector.logger.info('{} has been stopped!'.format(self.name))

        return True


    def run_ada(self):
        '''
            Run adaptive training preocess
        '''
        # 1. Process the dataset and save it:
        self.connector.logger.info('ADA: Preprocessing data...')
        # self.segment_dataset(self.dataset)

        # 2. Run adaptive training:
        self.connector.logger.info('ADA: Training...')
        clear_dir(self.conf['lwgan_ada']['checkpoints_dir'])
        args = TrainOptions().parse(params=self.conf['lwgan_ada'], set_cuda_env=False, verbose=False)
        trainer = LWGANTrainer(args)


    def segment_dataset(self, dataset):
        # Init segmentator:
        self.connector.logger.info('Initializing SegmentatorRT from {}'.format(
            self.conf['segmentator']['maskrcnn_path']))
        segmentator = SegmentatorRT(self.conf['segmentator'])

        # Make dataset dirs:
        imgs_dir = os.path.join(self.conf['lwgan_ada']['data_dir'], 'imgs')
        smpls_dir = os.path.join(self.conf['lwgan_ada']['data_dir'], 'smpls')
        clear_dir(imgs_dir)
        clear_dir(smpls_dir)
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        if not os.path.exists(smpls_dir):
            os.makedirs(smpls_dir)

        # Process and save:
        for t in tqdm(range(len(dataset))):
            _, masked_img = segmentator.inference(dataset[t]['lwgan_input_img'], apply_mask=True)

            img_path = os.path.join(imgs_dir, str(t).zfill(5) + '.jpeg')
            cv_utils.save_cv2_img(masked_img, img_path, normalize=True)
            smpl_path = os.path.join(smpls_dir, str(t).zfill(5) + '.pkl')
            write_pickle_file(smpl_path, dataset[t]['smpl_vec'])

        self.connector.logger.info('Dataset has been created! Total size: {}'.format(len(dataset)))


def run_live_stream():
    output_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/rt/holoport/adaptive_training'
    stream = LiveStream(output_dir)
    stream.run_model(HoloportAdaModel, path_to_conf=path_to_conf, label='holoport_adaptive_training')

def run_video_stream(output_dir):
    output_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/rt/holoport/adaptive_training'
    stream = VideoStream(output_dir, out_fps=20, skip_each_i_frame=3)
    stream.run_model(HoloportAdaModel, path_to_conf=path_to_conf, label='holoport_adaptive_training')

def main(path_to_conf):
    run_live_stream()
    # run_video_stream()

if __name__ == '__main__':
    path_to_conf = 'adaptive_training.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    main(path_to_conf)


'''
class HoloportBatchModel(object):
    LABEL = ['holoport_live', 'holoport_andrey', 'holoport_yulia']  # ignore the model
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None, path_to_conf='yolo-vibe-lwgan_live.yaml'):
        self.connector = connector
        self.connector.enable_frame_throw()
        self.name = 'HoloportBatchModel'

        # Load config:
        path_to_conf = os.path.join(os.path.dirname(__file__), path_to_conf)
        conf = parse_conf(path_to_conf)
        self.connector.logger.info('Config has been loaded from {}'.format(path_to_conf))

        # FPS:
        if 'target_fps' in conf['input']:
            self.target_fps = conf['input']['target_fps']
        else:
            self.target_fps = 15
        self.ms_per_frame = 1.0 / self.target_fps

        # Init model:
        self.yolo, self.yolo_args = init_yolo(conf['yolo'])
        self.vibe, self.vibe_args = init_vibe(conf['vibe'])
        self.lwgan, self.lwgan_args = init_lwgan(conf['lwgan'])
        if label is not None:
            if label == 'holoport_andrey':
                img_path = os.path.join(os.path.dirname(__file__), 'assets/andrey_260_img.tensor')
                smpl_path = os.path.join(os.path.dirname(__file__), 'assets/andrey_260_smpl.tensor')
                self.lwgan.load_descriptor(img_path,smpl_path)
                self.lwgan.desc_smpl = self.lwgan.desc_smpl[0]
            elif label == 'holoport_yulia':
                img_path = os.path.join(os.path.dirname(__file__), 'assets/yulia_166_img.tensor')
                smpl_path = os.path.join(os.path.dirname(__file__), 'assets/yulia_166_smpl.tensor')
                self.lwgan.load_descriptor(img_path,smpl_path)

        # Warmup:
        if 'warmup_img' in conf['input']:
            img = cv2.imread(conf['input']['warmup_img'], 1)
            warmup_holoport_pipeline(img, self.yolo, self.yolo_args,
                                     self.vibe, self.vibe_args,
                                     self.lwgan, self.lwgan_args)

        # Auxiliary params:
        self.aux_params = generate_aux_params(conf['input'])

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
        worker_args = (self.vibe_args, self.break_event, self.yolo_output_q, self.vibe_input_q, 0.005, True)
        self.workers.append(threading.Thread(target=pre_vibe_worker, args=worker_args))
        worker_args = (self.lwgan_args, self.break_event, self.vibe_output_q, self.lwgan_input_q)
        self.workers.append(threading.Thread(target=pre_lwgan_worker, args=worker_args))
        worker_args = (self.break_event, self.lwgan_output_q, self.avatar_q)
        self.workers.append(threading.Thread(target=postprocess_worker, args=worker_args))
        worker_args = (self.yolo, self.vibe, self.break_event, self.yolo_input_q,
                       self.yolo_output_q, self.vibe_input_q, self.vibe_output_q)
        self.workers.append(threading.Thread(target=yolo_vibe_inference_worker, args=worker_args))
        worker_args = (self.lwgan, self.break_event, self.lwgan_input_q, self.lwgan_output_q, 0.1, 2)
        self.workers.append(threading.Thread(target=lwgan_batch_inference_worker, args=worker_args))
        worker_args = (self.break_event, self.avatar_q, self.connector.send_data, self.connector.send_frame)
        self.workers.append(threading.Thread(target=send_worker, args=worker_args))

    def run(self):
        self.connector.logger.info('Running {}...'.format(self.name))

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
            # _ = self.connector.recv_all_frames()
            start = time.time()

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
'''
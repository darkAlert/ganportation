import cv2
import threading
from queue import Queue, Empty, Full
import time


def send_worker(break_event, frame_q, send_data, send_frame, timeout=0.005):
    print('send_worker has been run...')

    mean_latency = None
    mean_fps = None
    start = time.time()
    prev_frame_start = time.time()
    mean_wait = None

    while not break_event.is_set():
        try:
            data = frame_q.get(timeout=timeout)
            frame_q.task_done()
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

        #Draw:
        text = 'fps:{:.1f}'.format(mean_fps)
        cv2.putText(data['frame'], text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        text = 'lat:{:.1f}'.format(mean_latency*1000)
        cv2.putText(data['frame'], text, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        text = 'wait:{:.1f}'.format(mean_wait * 1000)
        cv2.putText(data['frame'], text, (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Send the result:
        send_data(dict(fps=mean_fps, latency=mean_latency, wait_time=mean_wait))
        send_frame(data['frame'])

    print('send_worker has been terminated.')

    return True


class StreamModel(object):
    LABEL = ['stream_realtime']
    SENDS_VIDEO = True
    SENDS_DATA = True

    def __init__(self, connector, label=None):
        self.connector = connector
        self.connector.enable_frame_throw()
        self.name = 'StreamModel'

        # Make queues, events and threads:
        self.break_event = threading.Event()
        self.frame_q = Queue(maxsize=10000)
        self.workers = []

        # Make workers:
        worker_args = (self.break_event, self.frame_q, self.connector.send_data, self.connector.send_frame)
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
                data = {'frame': frame.copy(), 'start': time.time()}
                self.frame_q.put(data, timeout=0.005)
            else:
                self.stop()

            # print('idx:{}, frames:{}'.format(idx, self.frame_q.qsize()))

    def stop(self):
        self.connector.logger.info('Stopping {}...'.format(self.name))
        self.break_event.set()

        for w in self.workers:
            w.join()

        self.connector.logger.info('{} has been stopped!'.format(self.name))

        return True
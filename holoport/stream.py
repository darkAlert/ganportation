import os
import cv2
import threading
import time
from lwganrt.utils.util import clear_dir


def get_paths(path_to_src):
    paths = []
    ext = ('.png', '.PNG', '.jpeg', '.JPG', '.jpeg', '.JPEG')
    for dirpath, dirnames, filenames in os.walk(path_to_src):
        for filename in [f for f in filenames if f.endswith(ext)]:
            paths.append(os.path.join(path_to_src, filename))
    paths.sort()

    return paths

class Logger():
    def __init__(self, name='Stream'):
        self.name = name
        pass

    def info(self, text):
        print ('[{}]: {}'.format(self.name, text))


class LiveStream():
    def __init__(self, output_dir=None):
        self.logger = Logger('LiveStream')
        self.model = None
        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(0)
        self.output_dir = output_dir

        if self.output_dir:
            clear_dir(self.output_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def run_model(self, ModelClass, **kwargs):
        self.counter = 0
        self.model = ModelClass(self, **kwargs)
        self.stop_event.clear()
        self.model.run()

    def recv_all_frames(self):
        return [None]

    def enable_frame_throw(self):
        return True

    def send_data(self, *args):
        return True

    def frames(self):
        while self.stop_event.is_set() == False:
            _, frame = self.cap.read()
            yield frame

        self.model.stop()
        cv2.destroyAllWindows()

        if self.output_dir:
            self.logger.info('Output frames have been saved to {}'.format(self.output_dir))
            self.logger.info('Total output frames={}'.format(self.counter))

    def send_frame(self, frame):
        if self.stop_event.is_set():
            return

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event.set()

        cv2.imshow('LiveStream', frame)

        if self.output_dir:
            name = str(self.counter).zfill(5) + '.jpeg'
            output_path = os.path.join(self.output_dir, name)
            cv2.imwrite(output_path, frame)
            self.counter += 1


class VideoStream():
    def __init__(self, source_dir, out_fps=30, skip_each_i_frame=None, output_dir=None):
        self.logger = Logger('VideoStream')
        self.model = None
        self.stop_event = threading.Event()
        self.cap = self._read_source(source_dir, skip_each_i_frame)
        self.fps = out_fps
        self.output_dir = output_dir
        if self.output_dir:
            clear_dir(self.output_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.logger.info('Frames have been read from {}'.format(source_dir))
        self.logger.info('Total frames={} with fps={}'.format(len(self.cap), self.fps))

    def _read_source(self, source_dir, skip_each_i_frame):
        paths = get_paths(source_dir)
        images = []
        for i, p in enumerate(paths):
            if skip_each_i_frame is not None and i % skip_each_i_frame == 0:
                continue
            images.append(cv2.imread(p,1))

        return images

    def run_model(self, ModelClass, **kwargs):
        self.counter = 0
        self.model = ModelClass(self, **kwargs)
        self.stop_event.clear()
        self.model.run()

    def recv_all_frames(self):
        return [None]

    def enable_frame_throw(self):
        return True

    def send_data(self, *args):
        return True

    def frames(self):
        idx = 0
        ms = 1.0 / self.fps
        start = time.time()

        while not self.stop_event.is_set():
            if idx >= len(self.cap):
                break

            # Rectify FPS:
            elapsed = time.time() - start
            if elapsed < ms:
                time.sleep(ms-elapsed)
            start = time.time()

            yield self.cap[idx]
            idx += 1

        self.model.stop()
        cv2.destroyAllWindows()

        if self.output_dir:
            self.logger.info('Output frames have been saved to {}'.format(self.output_dir))
            self.logger.info('Total output frames={}'.format(self.counter))

    def send_frame(self, frame):
        cv2.imshow('VideoStream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event.set()

        if self.output_dir:
            name = str(self.counter).zfill(5) + '.jpeg'
            output_path = os.path.join(self.output_dir, name)
            cv2.imwrite(output_path, frame)
            self.counter += 1


class VideoSaver():
    def __init__(self, output_dir, area_box=None):
        clear_dir(self.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pt1, pt2 = None, None
        if area_box is not None:
            pt1 = (area_box[0], area_box[1])
            pt2 = (area_box[0] + area_box[2], area_box[1] + area_box[3])

        cap = cv2.VideoCapture(0)
        counter = 0

        while True:
            _, frame = cap.read()

            # Save:
            name = str(counter).zfill(5) + '.jpeg'
            output_path = os.path.join(output_dir, name)
            cv2.imwrite(output_path, frame)
            counter += 1

            if area_box is not None:
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), thickness=2)

            cv2.imshow('VideoSaver', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print ('[VideoSaver] Frames has been saved to {}'.format(output_dir))
        print ('[VideoSaver] Total frames: {}'.format(counter))


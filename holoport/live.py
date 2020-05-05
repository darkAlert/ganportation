import cv2
import threading


class Logger():
    def __init__(self):
        pass

    def info(self, text):
        print ('[Live]:',text)


class LiveStream():
    def __init__(self):
        self.logger = Logger()
        self.cap = cv2.VideoCapture(0)
        self.model = None
        self.stop_event = threading.Event()

    def run_model(self, ModelClass, **kwargs):
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

    def send_frame(self, frame):
        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event.set()
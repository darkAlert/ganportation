import sys
import os
import cv2
import threading
from queue import Queue
from holoport.workers import *
from holoport.conf.conf_parser import parse_conf
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.hvibe import init_vibe
from holoport.hyolo import init_yolo
from holoport.tests.test_yolo_vibe_lwgan import load_frames


def test_multithreads(path_to_conf, save_results=True, realtime_ms=None):
    # Load configs:
    conf = parse_conf(path_to_conf)
    print('Config has been loaded from', path_to_conf)

    # Init YOLO-RT model:
    conf['yolo']['gpu_id'] = '1'
    yolo, yolo_args = init_yolo(conf['yolo'])

    # Init VIBE-RT model:
    conf['vibe']['gpu_id'] = '1'
    vibe, vibe_args = init_vibe(conf['vibe'])

    # Init LWGAN-RT model:
    conf['lwgan']['gpu_ids'] = '0'
    lwgan, lwgan_args = init_lwgan(conf['lwgan'])

    # Warmup:
    if 'warmup_img' in conf['input']:
        img = cv2.imread(conf['input']['warmup_img'],1)
        warmup_holoport_pipeline(img, yolo, yolo_args, vibe, vibe_args, lwgan, lwgan_args)

    # Load test data:
    print('Loading test data...')
    frames_dir = os.path.join(conf['input']['frames_dir'], conf['input']['target_path'])
    n = int(conf['input']['max_frames']) if 'max_frames' in conf['input'] else None
    test_data = load_frames(frames_dir, max_frames=n)
    print('Test data has been loaded:', len(test_data))

    # Avatar view params:
    steps = conf['input']['steps']
    view = parse_view_params(conf['input']['view'])

    # Dummy scene params:
    t = conf['input']['scene_bbox'].split(',')
    assert len(t) == 4
    dummy_scene_bbox = np.array([[int(t[0]), int(t[1]), int(t[2]), int(t[3])]], dtype=np.int64)
    dummy_scene_cbbox = dummy_scene_bbox.copy()
    dummy_scene_cbbox[:,0] = dummy_scene_bbox[:,0] + dummy_scene_bbox[:,2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
    dummy_scene_cbbox[:,1] = dummy_scene_bbox[:,1] + dummy_scene_bbox[:,3] * 0.5

    # Set auxiliary params:
    aux_params = {}
    aux_params['scene_bbox'] = dummy_scene_bbox
    aux_params['scene_cbbox'] = dummy_scene_cbbox
    aux_params['steps'] = steps
    aux_params['view'] = view

    # Make queues, events and threads:
    break_event = threading.Event()
    frame_q = Queue(maxsize=10000)
    yolo_input_q = Queue(maxsize=1000)
    yolo_output_q = Queue(maxsize=1000)
    vibe_input_q = Queue(maxsize=1000)
    vibe_output_q = Queue(maxsize=1000)
    lwgan_input_q = Queue(maxsize=1000)
    lwgan_output_q = Queue(maxsize=1000)
    avatar_q = Queue(maxsize=10000)
    workers = []

    # Make workers:
    worker_args = (yolo_args, break_event, frame_q, yolo_input_q, aux_params)
    workers.append(threading.Thread(target=pre_yolo_worker, args=worker_args))
    worker_args = (vibe_args, break_event, yolo_output_q, vibe_input_q)
    workers.append(threading.Thread(target=pre_vibe_worker, args=worker_args))
    worker_args = (lwgan_args, break_event, vibe_output_q, lwgan_input_q)
    workers.append(threading.Thread(target=pre_lwgan_worker, args=worker_args))
    worker_args = (break_event, lwgan_output_q, avatar_q)
    workers.append(threading.Thread(target=postprocess_worker, args=worker_args))
    worker_args = (yolo, vibe, break_event, yolo_input_q, yolo_output_q, vibe_input_q, vibe_output_q)
    workers.append(threading.Thread(target=yolo_vibe_inference_worker, args=worker_args))
    if realtime_ms is None:
        worker_args = (lwgan, break_event, lwgan_input_q, lwgan_output_q)
    else:
        worker_args = (lwgan, break_event, lwgan_input_q, lwgan_output_q, 0.005, True)
    workers.append(threading.Thread(target=lwgan_inference_worker, args=worker_args))

    # Feed data:
    if realtime_ms is None:
        for data in test_data:
            frame_q.put(data)

    print('Inferencing... realtime fps:', realtime_ms)
    start = time.time()

    # Run workers:
    for w in workers:
        w.start()

    if realtime_ms is not None:
        # Simulate real-time frame capturing:
        for data in test_data:
            frame_q.put(data)
            print('{}/{}, yolo_in:{}, yolo_out:{}, vibe_in:{}, vibe_out:{}, lwgan_in:{}, lwgan_out:{}'.format(
                avatar_q.qsize(), len(test_data), yolo_input_q.qsize(),
                yolo_output_q.qsize(), vibe_input_q.qsize(), vibe_output_q.qsize(),
                lwgan_input_q.qsize(), lwgan_output_q.qsize()))
            time.sleep(realtime_ms)

    else:
        # Wait for all the data to be processed
        while not frame_q.empty() or \
                not yolo_input_q.empty() or \
                not yolo_output_q.empty() or \
                not vibe_input_q.empty() or \
                not vibe_output_q.empty() or \
                not lwgan_input_q.empty() or \
                not lwgan_output_q.empty():
            print ('{}/{}, yolo_in:{}, yolo_out:{}, vibe_in:{}, vibe_out:{}, lwgan_in:{}, lwgan_out:{}'.format(
                avatar_q.qsize(), len(test_data), yolo_input_q.qsize(),
                yolo_output_q.qsize(), vibe_input_q.qsize(), vibe_output_q.qsize(),
                lwgan_input_q.qsize(), lwgan_output_q.qsize()))
            time.sleep(0.1)

    # Stop workers:
    break_event.set()

    # Wait workers:
    for w in workers:
        w.join()

    # Log:
    elapsed = time.time() - start
    n = len(test_data)
    m = avatar_q.qsize()
    fps = n / elapsed
    spf = elapsed / len(test_data)  # seconds per frame
    print('###Elapsed time:', elapsed, 'processed:{}/{}'.format(m,n), 'fps:', fps, 'spf:', spf)

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        print ('Saving the results to', result_dir)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        idx = 0

        while True:
            try:
                data = avatar_q.get(timeout=1)
                avatar_q.task_done()
                out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
                if 'not_found' in data:
                    dummy_output = np.zeros((100, 100, 3), dtype=np.uint8)
                    cv2.imwrite(out_path, dummy_output)
                else:
                    cv2.imwrite(out_path, data['avatar'])
                idx += 1
            except Empty:
                break

    print ('All done!')


def main():
    path_to_conf = 'holoport/conf/local/yolo_vibe_lwgan_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    test_multithreads(path_to_conf, realtime_ms=0.06)  # ~16.5 fps
    # test_multithreads(path_to_conf)

if __name__ == '__main__':
    main()
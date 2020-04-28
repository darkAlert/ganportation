import sys
import os
import time
import numpy as np
import cv2
import threading
from queue import Queue, Empty

from holoport.conf.conf_parser import parse_conf
from holoport.hlwgan import init_lwgan, parse_view_params
from holoport.hvibe import init_vibe
from holoport.hyolo import init_yolo
from holoport.tests.test_yolo_vibe_lwgan import pre_yolo, post_yolo, pre_vibe, post_vibe, pre_lwgan, post_lwgan
from holoport.tests.test_yolo_vibe_lwgan import load_frames


def pre_yolo_by_worker(args, break_event, input_q, output_q, aux_params):
    print('pre_yolo_by_worker has been run...')

    # Parse auxiliary params:
    dummy_scene_bbox = aux_params['dummy_scene_bbox']
    dummy_scene_cbbox = aux_params['dummy_scene_cbbox']
    steps = aux_params['steps']
    view = aux_params['view']
    delta = 360 / steps        # view changing params
    step_i = 0

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=0.005)
            input_q.task_done()
        except Empty:
            continue

        # Update avatar view:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0
        data['lwgan_input_view'] = view

        # Set scene bbox and cbbox:
        data['scene_bbox'] = dummy_scene_bbox
        data['scene_cbbox'] = dummy_scene_cbbox

        step_i += 1
        if step_i >= steps:
            step_i = 0

        # Prepare YOLO input:
        data = pre_yolo(data, args)
        output_q.put(data)

    print('pre_yolo_by_worker has been terminated.')

    return True


def pre_vibe_by_worker(args, break_event, input_q, output_q):
    print('pre_vibe_by_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=0.005)
            input_q.task_done()
        except Empty:
            continue

        # Prepare YOLO output:
        data = post_yolo(data)

        if data['yolo_cbbox'] is None:
            data['not_found'] = True
            output_q.put(data)
            print ('Skip frame {}: person not found!')
            continue

        # Prepare VIBE input:
        data = pre_vibe(data, args)
        output_q.put(data)

    print('pre_vibe_by_worker has been terminated.')

    return True


def pre_lwgan_by_worker(args, break_event, input_q, output_q):
    print('pre_lwgan_by_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=0.005)
            input_q.task_done()
        except Empty:
            continue

        if 'not_found' in data:
            output_q.put(data)
            continue

        # Prepare VIBE output:
        data = post_vibe(data)

        # Prepare LWGAN input:
        data = pre_lwgan(data, args)
        output_q.put(data)

    print('pre_lwgan_by_worker has been terminated.')

    return True


def postprocess_by_worker(break_event, input_q, output_q):
    print('postprocess_by_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=0.005)
            input_q.task_done()
        except Empty:
            continue

        if 'not_found' in data:
            output_q.put(data)
            continue

        # Prepare LWGAN output:
        data = post_lwgan(data)
        output_q.put(data)

    print('postprocess_by_worker has been terminated.')

    return True


def yolo_vibe_inference_by_worker(yolo, vibe, break_event, yolo_input_q, yolo_output_q, vibe_input_q, vibe_output_q):
    print('yolo_vibe_inference_by_worker has been run...')

    while not break_event.is_set():
        try:
            data = yolo_input_q.get(timeout=0.005)
            yolo_input_q.task_done()

            # YOLO inference:
            data['yolo_output'] = yolo.inference(data['yolo_input'])
            yolo_output_q.put(data)
        except Empty:
            pass

        try:
            data = vibe_input_q.get(timeout=0.005)
            vibe_input_q.task_done()

            if 'not_found' in data:
                vibe_output_q.put(data)
            else:
                # YOLO inference:
                data['vibe_output'] = vibe.inference(data['vibe_input'])
                vibe_output_q.put(data)
        except Empty:
            pass

    print('yolo_vibe_inference_by_worker has been terminated.')

    return True


def lwgan_inference_by_worker(lwgan, break_event, input_q, output_q):
    print('lwgan_inference_by_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=0.005)
            input_q.task_done()
        except Empty:
            continue

        if 'not_found' in data:
            output_q.put(data)
            continue

        # LWGAN inference:
        data['lwgan_output'] = lwgan.inference(data['lwgan_input_img'],
                                               data['lwgan_input_smpl'],
                                               data['lwgan_input_view'])
        output_q.put(data)

    print('lwgan_inference_by_worker has been terminated.')

    return True


def test_yolo_vibe_lwgan_multithreads(path_to_conf, save_results=True):
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
    aux_params['dummy_scene_bbox'] = dummy_scene_bbox
    aux_params['dummy_scene_cbbox'] = dummy_scene_cbbox
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

    # Make pre yolo worker:
    worker_args = (yolo_args, break_event, frame_q, yolo_input_q, aux_params)
    workers.append(threading.Thread(target=pre_yolo_by_worker, args=worker_args))

    # Make pre vibe worker:
    worker_args = (vibe_args, break_event, yolo_output_q, vibe_input_q)
    workers.append(threading.Thread(target=pre_vibe_by_worker, args=worker_args))

    # Make pre lwgan worker:
    worker_args = (lwgan_args, break_event, vibe_output_q, lwgan_input_q)
    workers.append(threading.Thread(target=pre_lwgan_by_worker, args=worker_args))

    # Make postprocess worker:
    worker_args = (break_event, lwgan_output_q, avatar_q)
    workers.append(threading.Thread(target=postprocess_by_worker, args=worker_args))

    # Make yolo+vibe worker:
    worker_args = (yolo, vibe, break_event, yolo_input_q, yolo_output_q, vibe_input_q, vibe_output_q)
    workers.append(threading.Thread(target=yolo_vibe_inference_by_worker, args=worker_args))

    # Make lwgan worker:
    worker_args = (lwgan, break_event, lwgan_input_q, lwgan_output_q)
    workers.append(threading.Thread(target=lwgan_inference_by_worker, args=worker_args))

    # Feed data:
    for data in test_data:
        frame_q.put(data)

    print('Inferencing...')
    start = time.time()

    # Run workers:
    for w in workers:
        w.start()

    # Wait for all the data to be processed
    while not frame_q.empty() or \
            not yolo_input_q.empty() or \
            not yolo_output_q.empty() or \
            not vibe_input_q.empty() or \
            not vibe_output_q.empty() or \
            not lwgan_input_q.empty() or \
            not lwgan_output_q.empty():
        print ('{}/{}, yolo_in:{}, yolo_out:{}, vibe_in:{}, vibe_out:{}, lwgan_in:{}, lwgan_out:{}'.format(avatar_q.qsize(), len(test_data), yolo_input_q.qsize(), yolo_output_q.qsize(), vibe_input_q.qsize(), vibe_output_q.qsize(), lwgan_input_q.qsize(), lwgan_output_q.qsize()))
        time.sleep(0.1)

    # Stop workers:
    break_event.set()

    # Wait workers:
    for w in workers:
        w.join()

    # Log:
    elapsed = time.time() - start
    n = len(test_data)
    fps = n / elapsed
    spf = elapsed / len(test_data)  # seconds per frame
    print('###Elapsed time:', elapsed, 'frames:', n, 'fps:', fps, 'spf:', spf)

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

    test_yolo_vibe_lwgan_multithreads(path_to_conf)

if __name__ == '__main__':
    main()
import numpy as np
from queue import Empty
import time
from holoport.hlwgan import pre_lwgan, post_lwgan, parse_view_params
from holoport.hvibe import pre_vibe, post_vibe
from holoport.hyolo import pre_yolo, post_yolo


def warmup_holoport_pipeline(img, yolo, yolo_args, vibe, vibe_args, lwgan, lwgan_args):
    print('Warming up holoport pipeline...')
    assert img is not None

    # Set dummy input:
    data = {'frame': img}
    view = parse_view_params('R=0,90,0/t=0,0,0')
    view['R'][0] = 0
    view['R'][1] = 0
    view['R'][2] = 0
    data['lwgan_input_view'] = view
    dummy_scene_bbox = np.array([[575, 150, 850, 850]], dtype=np.int64)
    dummy_scene_cbbox = dummy_scene_bbox.copy()
    dummy_scene_cbbox[:, 0] = dummy_scene_bbox[:, 0] + dummy_scene_bbox[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
    dummy_scene_cbbox[:, 1] = dummy_scene_bbox[:, 1] + dummy_scene_bbox[:, 3] * 0.5
    data['scene_bbox'] = dummy_scene_bbox
    data['scene_cbbox'] = dummy_scene_cbbox

    # YOLO:
    data = pre_yolo(data, yolo_args)
    data['yolo_output'] = yolo.inference(data['yolo_input'])
    data = post_yolo(data)

    assert data['yolo_cbbox'] is not None

    # VIBE:
    data = pre_vibe(data, vibe_args)
    data['vibe_output'] = vibe.inference(data['vibe_input'])
    data = post_vibe(data)

    # LWGAN:
    data = pre_lwgan(data, lwgan_args)
    data['lwgan_output'] = lwgan.inference(data['lwgan_input_img'],
                                           data['lwgan_input_smpl'],
                                           data['lwgan_input_view'])

    return True


def pre_yolo_worker(args, break_event, input_q, output_q, aux_params, timeout=0.005):
    print('pre_yolo_worker has been run...')

    # Parse auxiliary params:
    dummy_scene_bbox = aux_params['dummy_scene_bbox']
    dummy_scene_cbbox = aux_params['dummy_scene_cbbox']
    steps = aux_params['steps']
    view = aux_params['view']
    delta = 360 / steps        # view changing params
    step_i = 0

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
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

    print('pre_yolo_worker has been terminated.')

    return True


def pre_vibe_worker(args, break_event, input_q, output_q, timeout=0.005):
    print('pre_vibe_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Prepare YOLO output:
        data = post_yolo(data)

        if data['yolo_cbbox'] is None:
            data['not_found'] = True
            output_q.put(data)
            continue

        # Prepare VIBE input:
        data = pre_vibe(data, args)
        output_q.put(data)

    print('pre_vibe_worker has been terminated.')

    return True


def pre_lwgan_worker(args, break_event, input_q, output_q, timeout=0.005):
    print('pre_lwgan_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
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

    print('pre_lwgan_worker has been terminated.')

    return True


def postprocess_worker(break_event, input_q, output_q, timeout=0.005):
    print('postprocess_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        if 'not_found' in data:
            output_q.put(data)
            continue

        # Prepare LWGAN output:
        data = post_lwgan(data)
        output_q.put(data)

    print('postprocess_worker has been terminated.')

    return True


def yolo_vibe_inference_worker(yolo, vibe, break_event, yolo_input_q, yolo_output_q, vibe_input_q, vibe_output_q, timeout1=0.005, timeout2=0.005):
    print('yolo_vibe_inference_worker has been run...')

    while not break_event.is_set():
        try:
            data = yolo_input_q.get(timeout=timeout1)
            yolo_input_q.task_done()

            # YOLO inference:
            data['yolo_output'] = yolo.inference(data['yolo_input'])
            yolo_output_q.put(data)
        except Empty:
            pass

        try:
            data = vibe_input_q.get(timeout=timeout2)
            vibe_input_q.task_done()

            if 'not_found' in data:
                vibe_output_q.put(data)
            else:
                # YOLO inference:
                data['vibe_output'] = vibe.inference(data['vibe_input'])
                vibe_output_q.put(data)
        except Empty:
            pass

    print('yolo_vibe_inference_worker has been terminated.')

    return True


def lwgan_inference_worker(lwgan, break_event, input_q, output_q, timeout=0.005, skip_frames=False):
    print('lwgan_inference_worker has been run...')

    while not break_event.is_set():
        if skip_frames:
            data = None
            try:
                while True:
                    data = input_q.get_nowait()
                    input_q.task_done()
            except Empty:
                if data is None:
                    time.sleep(timeout)
                    continue
        else:
            try:
                data = input_q.get(timeout=timeout)
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

    print('lwgan_inference_worker has been terminated.')

    return True
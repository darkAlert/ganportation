import numpy as np
from queue import Empty
import time
from holoport.hlwgan import pre_lwgan, post_lwgan, parse_view_params
from holoport.hvibe import pre_vibe, post_vibe
from holoport.hyolo import pre_yolo, post_yolo


def box_fits_into_scene(bbox, scene_bbox):
    '''
    Check if the predicted yolo box fits into the scene box
    '''
    scence_x1, scence_x2 = scene_bbox[0], scene_bbox[0] + scene_bbox[2]
    scence_y1, scence_y2 = scene_bbox[1], scene_bbox[1] + scene_bbox[3]
    x = bbox[0]+int(round(bbox[2]*0.5))
    y1, y2 = bbox[1], bbox[1] + bbox[3]

    if y1 < scence_y1 or y2 > scence_y2:
        return False
    if x < scence_x1 or x > scence_x2:
        return False

    return True


def warmup_holoport_pipeline(img, yolo, yolo_args, vibe=None, vibe_args=None, lwgan=None, lwgan_args=None):
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
    print('YOLO has been warmed up!')

    assert data['yolo_cbbox'] is not None

    # VIBE:
    if vibe is None or vibe_args is None:
        return True
    data = pre_vibe(data, vibe_args)
    data['vibe_output'] = vibe.inference(data['vibe_input'])
    data = post_vibe(data)
    print('VIBE has been warmed up!')

    # LWGAN:
    if lwgan is None or lwgan_args is None:
        return True
    data = pre_lwgan(data, lwgan_args)
    data['lwgan_output'] = lwgan.inference(data['lwgan_input_img'],
                                           data['lwgan_input_smpl'],
                                           data['lwgan_input_view'])
    print('LWGAN has been warmed up!')

    return True


def pre_yolo_worker(args, break_event, input_q, output_q, aux_params, timeout=0.005):
    print('pre_yolo_worker has been run...')

    # Parse auxiliary params:
    scene_bbox = aux_params['scene_bbox']
    scene_cbbox = aux_params['scene_cbbox']

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        # Set scene bbox and cbbox:
        data['scene_bbox'] = scene_bbox
        data['scene_cbbox'] = scene_cbbox

        # Prepare YOLO input:
        data = pre_yolo(data, args)
        output_q.put(data)

    print('pre_yolo_worker has been terminated.')

    return True


def pre_vibe_worker(args, break_event, input_q, output_q, timeout=0.005, scene_fitting=False):
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

        if scene_fitting:
            if not box_fits_into_scene(data['yolo_bbox'][0], data['scene_bbox'][0]):
                data['not_found'] = True
                output_q.put(data)
                continue

        # Prepare VIBE input:
        data = pre_vibe(data, args)
        output_q.put(data)

    print('pre_vibe_worker has been terminated.')

    return True


def pre_lwgan_worker(args, break_event, input_q, output_q, aux_params, timeout=0.005):
    print('pre_lwgan_worker has been run...')

    # View changing params:
    steps = aux_params['steps']
    view = aux_params['view']
    delta = 360 / steps
    step_i = 0

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

        # Update avatar view:
        view['R'][0] = 0
        view['R'][1] = delta * step_i / 180.0 * np.pi
        view['R'][2] = 0
        if 'new_R' in data:
            view['R'][1] = data['new_R'] / 180.0 * np.pi
        data['lwgan_input_view'] = view

        # Prepare LWGAN input:
        data = pre_lwgan(data, args)
        output_q.put(data)

        step_i += 1
        if step_i >= steps:
            step_i = 0


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

def lwgan_inference_worker(lwgan, break_event, input_q, output_q, timeout=0.005, skip_frames=False, adaptive=None):
    print('lwgan_inference_worker has been run...')
    personalized = False

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

        if adaptive and not personalized:
            import cv2
            from lwganrt.utils.cv_utils import transform_img

            data_out = data.copy()
            img = cv2.cvtColor(data_out['frame'], cv2.COLOR_BGR2RGB)
            bbox = data_out['scene_bbox'][0]
            img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            text = 'ADAPTIVE PERSONALIZATION...'
            cv2.putText(img, text, (5, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            data_out['lwgan_output'] = transform_img(img, 256) * 2 - 1.0
            output_q.put(data_out.copy())
            time.sleep(0.1)
            output_q.put(data_out)

            lwgan.run_adaptive_personalization(data['lwgan_input_img'],
                                               data['lwgan_input_smpl'])
            personalized = True
            continue

        # LWGAN inference:
        data['lwgan_output'] = lwgan.inference(data['lwgan_input_img'],
                                               data['lwgan_input_smpl'],
                                               data['lwgan_input_view'])

        output_q.put(data)

    print('lwgan_inference_worker has been terminated.')

    return True

def lwgan_batch_inference_worker(lwgan, break_event, input_q, output_q, timeout=0.001, batch_size=2):
    '''
    NOT IMPLEMENTED YET ...
    '''
    raise NotImplementedError
    print('lwgan_batch_inference_worker has been run...')

    while not break_event.is_set():
        data_batch = []

        for i in range(batch_size):
            try:
                data = input_q.get(timeout=timeout)
                input_q.task_done()
                data_batch.append(data)
            except Empty:
                pass

        # LWGAN batch inference:
        if len(data_batch) > 0:
            frames = [data['lwgan_input_img'] for data in data_batch if 'not_found' not in data]
            smpls = [data['lwgan_input_smpl'] for data in data_batch if 'not_found' not in data]
            if len(frames) > 0:
                preds = lwgan.inference_batch(frames, smpls)
            else:
                preds = []
            idx = 0
            for data in data_batch:
                if 'not_found' not in data:
                    data['lwgan_output'] = preds[idx]
                    idx += 1
                output_q.put(data)

    print('lwgan_batch_inference_worker has been terminated.')

    return True


def segm_inference_worker(segmentator, break_event, input_q, output_q, timeout=0.005):
    print('segm_inference_worker has been run...')

    while not break_event.is_set():
        try:
            data = input_q.get(timeout=timeout)
            input_q.task_done()
        except Empty:
            continue

        if 'not_found' in data:
            output_q.put(data)
            continue

        # Segmentator inference:
        _, masked_img = segmentator.inference(data['lwgan_input_img'], apply_mask=True)
        data['masked_img'] = masked_img

        output_q.put(data)

    print('segm_inference_worker has been terminated.')

    return True
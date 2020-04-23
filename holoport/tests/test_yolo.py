import sys
import os
import cv2
import time
from holoport.hyolo import init_yolo, prepare_yolo_input, convert_yolo_output_to_bboxes
from holoport.conf.conf_parser import parse_conf


def get_file_paths(path, exts=('.jpeg','.jpg','.png')):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if any(f.endswith(ext) for ext in exts)]:
            file_paths.append(os.path.join(dirpath,filename))
            file_paths.sort()

    return file_paths


def load_data(frames_dir, img_size):
    # Load frames:
    frame_paths = get_file_paths(frames_dir)
    images = []
    for path in frame_paths:
        images.append(cv2.imread(path,1))

    # Prepare data:
    data = []
    for img in images:
        prep_img = prepare_yolo_input(img.copy(), img_size)
        data.append({'yolo_input': prep_img, 'origin_frame': img})

    return data


def test(path_to_conf, save_results=False):
    # Load config:
    conf = parse_conf(path_to_conf)
    print ('Config has been loaded from', path_to_conf)

    # Init YOLO-RT model:
    conf['yolo']['gpu_id'] = '0'
    yolo, args = init_yolo(conf['yolo'])

    # Load test data:
    print('Loading test data...')
    frames_dir = os.path.join(conf['input']['frames_dir'], conf['input']['target_path'])
    test_data = load_data(frames_dir, args.yolo_img_size)
    print('Test data has been loaded:', len(test_data))

    # Inference:
    print('Inferencing...')
    start = time.time()

    for data in test_data:
        output = yolo.inference(data['yolo_input'])
        data['yolo_output'] = output

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    spf = elapsed / len(test_data)  # secons per frame
    print('###Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps, 'spf:', spf)

    # Prepare output:
    for data in test_data:
        actual_size = [data['yolo_input'].shape[2:]]
        origin_size = [data['origin_frame'].shape]
        bboxes = convert_yolo_output_to_bboxes(data['yolo_output'], actual_size, origin_size)
        data['yolo_bbox'] = bboxes[0] if len(bboxes) else None

    # Save the results:
    result_dir = conf['output']['result_dir']
    if save_results and result_dir is not None:
        print ('Saving the results to', result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for idx, data in enumerate(test_data):
            if data['yolo_bbox'] is not None:
                pt1 = data['yolo_bbox'][0], data['yolo_bbox'][1]
                pt2 = data['yolo_bbox'][2], data['yolo_bbox'][3]
                img = cv2.rectangle(data['origin_frame'], pt1, pt2, color=(0,0,255))
                out_path = os.path.join(result_dir, str(idx).zfill(5) + '.jpeg')
                cv2.imwrite(out_path, img)
            # print ('{}: {}'.format(idx,data['yolo_bbox']))

    print ('All done!')


if __name__ == '__main__':
    path_to_conf = 'holoport/conf/local/yolo_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    test(path_to_conf, save_results=True)

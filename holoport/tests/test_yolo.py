import sys
import os
import cv2
import time
from holoport.hyolo import init_yolo, prepare_yolo_input
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
        prep_img = prepare_yolo_input(img, img_size)
        data.append({'yolo_input': prep_img})

    return data


def main(path_to_conf):
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
    results = []
    start = time.time()

    for data in test_data:
        output = yolo.inference(data['yolo_input'])
        results.append(output)

    elapsed = time.time() - start
    fps = len(test_data) / elapsed
    spf = elapsed/len(test_data)  #secons per frame
    print('Elapsed time:', elapsed, 'frames:', len(test_data), 'fps:', fps, 'spf:', spf)

    # Save the results:
    result_dir = conf['output']['result_dir']
    # for idx, res in enumerate(results):
    #     print ('{}: {}'.format(idx,res))

    print ('All done!')


if __name__ == '__main__':
    path_to_conf = 'holoport/conf/local/yolo_conf_local.yaml'
    if len(sys.argv) > 1:
        path_to_conf = sys.argv[1]
        sys.argv = [sys.argv[0]]

    main(path_to_conf=path_to_conf)

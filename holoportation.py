import os
import argparse
import time
from hvibe import HoloVibeRT, load_data

def init_vibe():
    print('Initializing VIBE...')

    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help='repo root dir')
    parser.add_argument('--smpl_model_dir', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data',
                        help='dir the containing SMPL model')
    parser.add_argument('--smpl_mean_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/smpl_mean_params.npz',
                        help='path to SMPL mean params file')
    parser.add_argument('--j_regressor_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/J_regressor_extra.npy',
                        help='path to Joint regressor model')
    parser.add_argument('--spin_model_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/spin_model_checkpoint.pth.tar',
                        help='path to spin model')
    parser.add_argument('--vibe_model_path', type=str,
                        default='thirdparty/vibe/vibert/data/vibe_data/vibe_model_wo_3dpw.pth.tar',
                        help='path to pretrained VIBE model')
    parser.add_argument('--seqlen', type=int, default=1,
                        help='VIBE sequence length')
    parser.add_argument('--bbox_scale', type=float, default=1.1,
                        help='scale for input bounding box')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size for input image')
    args = parser.parse_args()

    return HoloVibeRT(args)


def load_test_data():
    print('Loading test data...')

    frames_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames'
    yolo_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes_by_maskrcnn'
    avatar_bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes'
    target_path = 'person_2/light-100_temp-5600/garments_2/front_position/cam1'
    bbox_scale = 1.1
    crop_size = 224

    frames, yolo_bboxes, avatar_bboxes, frame_paths = \
        load_data(frames_dir, yolo_bboxes_dir, avatar_bboxes_dir, target_path, bbox_scale, crop_size)

    print('Test data has been loaded:', frames.shape)

    return frames, yolo_bboxes, avatar_bboxes, frame_paths



def test_vibe():
    # Init VIBE-RT model:
    vibe = init_vibe()

    # Load test data:
    frames, yolo_bboxes, avatar_bboxes, frame_paths = load_test_data()

    # Inference:
    print('Inferencing...')
    vibe_outputs = []
    start = time.time()

    for frame in frames:
        output = vibe.inference(frame)
        vibe_outputs.append(output)

    elapsed = time.time() - start
    fps = frames.shape[0] / elapsed
    print('Elapsed time:', elapsed, 'frames:', frames.shape[0], 'fps:', fps)


def main():
    test_vibe()

    print('All done!')


if __name__ == '__main__':
    main()
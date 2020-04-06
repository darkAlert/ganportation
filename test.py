import os
import argparse
# from thirdparty.vibe_lwgan.models.vibe_rt import VibeRT
# from thirdparty.vibe_lwgan.vibe_lwgan.lib.models.vibe_rt import VibeRT
from vibe_lwgan.lib.models.vibe_rt import VibeRT


def main():
    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    args = parser.parse_args()

    # vibe = VibeRT(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True)

    # print (demo)

if __name__ == '__main__':
    main()
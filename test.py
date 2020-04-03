import os
import argparse

import importlib.util
spec = importlib.util.spec_from_file_location('VibeRT', '/home/darkalert/builds/vibe/lib/models/vibe_rt.py')
VibeRT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VibeRT)



def main():
    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    args = parser.parse_args()

    vibe = VibeRT(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True)

if __name__ == '__main__':
    main()
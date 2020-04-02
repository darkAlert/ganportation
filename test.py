import os
import argparse



def main():
    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    args = parser.parse_args()

if __name__ == '__main__':
    main()
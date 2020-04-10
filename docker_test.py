import torch
import os

def main():
    print('Hello world!')
    x = torch.rand(5, 3)
    print(x)
    print (torch.cuda.is_available())
    print (torch.device('cuda'))
    print (torch.cuda.current_device())
    print(torch.cuda.device_count())
    print (torch.cuda.get_device_name(0))

    os.system('nvidia-smi')


if __name__ == '__main__':
    main()
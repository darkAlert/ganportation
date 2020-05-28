
                                                                             
<p float="center">
  <img src="https://s7.gifyu.com/images/rotation19461b210adb08b6.gif" width="49%" />
  <img src="https://s5.gifyu.com/images/ezgif.com-optimize6d7c4d9d7251b20a.gif" width="49%" />
</p>

# USING DOCKER #


### Install Docker ###
```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
or, if you are having problems, you can try the following:
```
curl -fsSL test.docker.com | sh
```

### INSTALL NVIDIA CONTAINER ###
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Add requirements ###
```
Add the required libraries to the requirements.txt
```

### Build & run dockerfile containing HoloPortRT (YOLOv3-RT + VIBE-RT + LWGAN-RT) ###
```
sudo docker build -t holoport-cuda101:1.0.0 .
sudo docker run --gpus all --rm --ipc=host -it holoport-cuda101:1.0.0 python3 docker_test.py
```
or you can install it by running a bash script:
```
sh install.sh
```

### Build & run dockerfile containing FacialAnimation (RetinaFace-RT + AUs-RT) ###
```
sudo docker build -f Dockerfile-aus -t aus-cuda101:1.0.0 .
sudo docker run --gpus all --rm --ipc=host -it aus-cuda101:1.0.0 python3 docker_test.py
```
or you can install it by running a bash script:
```
sh install-aus.sh
```

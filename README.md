
<p float="center">
  <img src="https://s7.gifyu.com/images/rotation.gif" />
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

### Build & run dockerfile ###
```
sudo docker build -t docker-test .
sudo docker run --gpus all --rm --ipc=host -it docker-test python3 docker_test.py
```

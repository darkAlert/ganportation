﻿﻿﻿﻿# USING DOCKER #


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

### Build & run dockerfile for HoloPortationRT ###
```
sudo docker build -t holoport-cuda101:1.0.0 .
sudo docker run --gpus all --rm --ipc=host -it holoport-cuda101:1.0.0 python3 docker_test.py
```

### Build & run dockerfile for RetinaFace + AUs ###
```
sudo docker build -f Dockerfile-aus -t aus-cuda101:1.0.0 .
sudo docker run --gpus all --rm --ipc=host -it aus-cuda101:1.0.0 python3 docker_test.py
```

### Upload resources from Azure Blob Storage ###
1. Upload the required resources in .tar.gz format (models and data) to the Azure Blob Storage using the Azure Portal or the Storage-Explorer utility (```sudo snap install storage-explorer```)
2. Generate a shared access signature (SAS) for the resources (https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview)
3. Use the generated SAS in dockerfile (the resourses will be downloaded and unpacked by the AzCopy utility)
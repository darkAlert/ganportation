FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C TZ=UTC

# install some basic utilities:
RUN set -xue ;\
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        unzip \
        wget \
        git \
        python3-dev \
        python3-pip \
        python3-yaml \
        ninja-build \
    ;\
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip==20.0.2

# download AzCopy utility
RUN mkdir -p /usr/src/utils ;\
    wget -O /usr/src/utils/azcopy_linux.tar.gz https://aka.ms/downloadazcopy-v10-linux ;\
    tar -xvzf /usr/src/utils/azcopy_linux.tar.gz --strip 1 -C /usr/src/utils

# download vibe_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/vibe_data.tar.gz?sp=rl&st=2020-05-06T15:15:19Z&se=2020-06-30T15:15:00Z&sv=2019-10-10&sr=b&sig=UW8aiEEC3ZmReT%2FJ4x6DDGHmMAlGneb8jcO2M%2BMJCk8%3D" "/usr/src/models/vibe_data.tar.gz" ;\
    tar -xvf /usr/src/models/vibe_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/vibe_data.tar.gz

# download yolo_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/yolo_data.tar.gz?sp=rl&st=2020-05-06T15:18:10Z&se=2020-06-30T15:18:00Z&sv=2019-10-10&sr=b&sig=FLB4s5ZCWhWdxgXPxt%2FI1tExuvzDlNJbyZZ%2B1QVMDso%3D" "/usr/src/models/yolo_data.tar.gz" ;\
    tar -xvf /usr/src/models/yolo_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/yolo_data.tar.gz

# download lwgan_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/lwgan_data.tar.gz?sp=rl&st=2020-05-06T15:19:31Z&se=2020-06-30T15:19:00Z&sv=2019-10-10&sr=b&sig=dVgBC1SQ99A2uEDY%2F7LDTQLsRmZ3wlXMjB9pAjwuA88%3D" "/usr/src/models/lwgan_data.tar.gz" ;\
    tar -xvf /usr/src/models/lwgan_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/lwgan_data.tar.gz

# download holovideo from Azure Blob Storage
RUN mkdir -p /usr/src/data ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/data/holovideo.tar.gz?sp=rl&st=2020-05-06T15:20:15Z&se=2020-06-30T15:20:00Z&sv=2019-10-10&sr=b&sig=9mLRN%2Ftv%2FSxftJUrGt0W2yhegiV6zQUkAEtgSwLIZog%3D" "/usr/src/data/holovideo.tar.gz" ;\
    tar -xvf /usr/src/data/holovideo.tar.gz -C /usr/src/data ;\
    rm /usr/src/data/holovideo.tar.gz

# install pytorch and torchvision with cuda101:
RUN pip3 install \
    https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp36-cp36m-linux_x86_64.whl

# copy all files to the container
WORKDIR /usr/src/app
COPY . .

# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt ;\
    pip3 install --no-cache-dir -r thirdparty/yolo/requirements.txt ;\
    pip3 install --no-cache-dir -r thirdparty/vibe/requirements.txt ;\
    pip3 install --no-cache-dir -r thirdparty/impersonator/requirements.txt

# install HOLOPORT-RT
RUN python3 setup.py install

# install YOLO-RT
WORKDIR /usr/src/app/thirdparty/yolo
RUN python3 setup.py install

# install VIBE-RT
WORKDIR /usr/src/app/thirdparty/vibe
RUN python3 setup.py install

# install neural_renderer (for LWGAN-RT)
WORKDIR /usr/src/app/thirdparty/impersonator/lwganrt/thirdparty/neural_renderer
ENV TORCH_CUDA_ARCH_LIST=Volta;Pascal
RUN python3 setup.py install

# install LWGAN-RT
WORKDIR /usr/src/app/thirdparty/impersonator
RUN python3 setup.py install

# set a directory for the app
WORKDIR /usr/src/app

# make a subdir to save the results
RUN mkdir -p /usr/src/app/outputs

# run the command
CMD ["python3"]
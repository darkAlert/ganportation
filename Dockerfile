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

# download AzCopy utility
RUN mkdir -p /usr/src/utils ;\
    wget -O /usr/src/utils/azcopy_linux.tar.gz https://aka.ms/downloadazcopy-v10-linux ;\
    tar -xvzf /usr/src/utils/azcopy_linux.tar.gz --strip 1 -C /usr/src/utils

# download vibe_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/vibe_data.tar.gz?sp=r&st=2020-04-13T14:35:21Z&se=2020-04-30T22:35:21Z&spr=https&sv=2019-02-02&sr=b&sig=%2BLQV9Cl20Ly%2BH6TgApyCsVZwuh50cN3vOdX3Pjzdh%2BA%3D" "/usr/src/models/vibe_data.tar.gz" ;\
    tar -xvf /usr/src/models/vibe_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/vibe_data.tar.gz

# download yolo_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/yolo_data.tar.gz?st=2020-04-28T09%3A31%3A08Z&se=2020-05-31T09%3A31%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=n%2B%2FATNCqLJOZB2a794%2F3F%2BCz2a9BHzLTh4SgCUAqsZ0%3D" "/usr/src/models/yolo_data.tar.gz" ;\
    tar -xvf /usr/src/models/yolo_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/yolo_data.tar.gz

# download lwgan_data from Azure Blob Storage
RUN mkdir -p /usr/src/models ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/models/lwgan_data.tar.gz?st=2020-04-28T09%3A32%3A41Z&se=2020-05-31T09%3A32%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=p6rJqCOGY58r%2BtrgOHbTz%2BcKNWCpnuKOcjXw2xa0ct0%3D" "/usr/src/models/lwgan_data.tar.gz" ;\
    tar -xvf /usr/src/models/lwgan_data.tar.gz -C /usr/src/models ;\
    rm /usr/src/models/lwgan_data.tar.gz

# download holovideo from Azure Blob Storage
RUN mkdir -p /usr/src/data ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/data/holovideo.tar.gz?st=2020-04-28T13%3A46%3A06Z&se=2020-05-31T13%3A46%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=ABK8F7bdK4vuK2sHA3MMI1318P4VoVSaRBvNZaOGNAQ%3D" "/usr/src/data/holovideo.tar.gz" ;\
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
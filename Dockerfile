FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

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
        python3-pip \
        python3-yaml \
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

# download holovideo from Azure Blob Storage
RUN mkdir -p /usr/src/data ;\
    /usr/src/utils/azcopy copy "https://holoportation.blob.core.windows.net/holoportation-rt/data/holovideo.tar.gz?sp=r&st=2020-04-14T09:37:23Z&se=2020-04-30T17:37:23Z&spr=https&sv=2019-02-02&sr=b&sig=ZxUtmbx5GFQaeSTU3%2BPieGXE5nqBVR0N9ICo8Hr8rzg%3D" "/usr/src/data/holovideo.tar.gz" ;\
    tar -xvf /usr/src/data/holovideo.tar.gz -C /usr/src/data ;\
    rm /usr/src/data/holovideo.tar.gz

# install pytorch and torchvision with cuda10:
RUN pip3 install \
    https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp36-cp36m-linux_x86_64.whl

# copy all files to the container
WORKDIR /usr/src/app
COPY . .

# install dependencies
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt ;\
    pip3 install --upgrade --no-cache-dir -r thirdparty/vibe/requirements.txt

# install HOLOPORT-RT
RUN python3 setup.py install

# install VIBE-RT
WORKDIR /usr/src/app/thirdparty/vibe
RUN python3 setup.py install

# set a directory for the app
WORKDIR /usr/src/app

# make a subdir to save the results
RUN mkdir -p /usr/src/app/outputs

# run the command
CMD ["python3"]
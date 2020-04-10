FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# install some basic utilities:
RUN set -xue ;\
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        unzip \
        python3-pip \
    ;\
    apt-get clean

# set a directory for the app
WORKDIR /usr/src/app

# copy all files to the container
COPY . .

# install pytorch and torchvision with cuda10:
RUN pip3 install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt


# run the command
CMD ["python3"]
#! /bin/bash

root_dir=~/builds
img_name=holoport-cuda101:1.0.0
username=holoanton

mkdir -p ${root_dir}
cd ${root_dir}

# get password
stty -echo
read -p "Git password please: " password
stty echo
printf '\n'

# clone repo
git config --global credential.helper 'cache --timeout=600'
git clone https://${username}:${password}@bitbucket.org/kazendi/holoportrt.git holoportrt
cd holoportrt
git submodule update --init --recursive

# build
docker build -t ${img_name} .

# test
test_cmd='python3 holoport/tests/test_vibe.py holoport/conf/azure/vibe_conf_azure.yaml'
docker run --gpus all --rm --ipc=host -it ${img_name} ${test_cmd}
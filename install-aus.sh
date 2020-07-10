#! /bin/bash

root_dir=~/builds
img_name=aus-cuda101:1.0.0
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
printf 'Bulding...\n'
docker build -f Dockerfile-aus -t ${img_name} .

# test
printf 'Testing...\n'
docker run --gpus all --rm --ipc=host -it ${img_name} python3 docker_test.py

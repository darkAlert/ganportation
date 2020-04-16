#! /bin/bash

root_dir=~/builds
name=holoport-cuda101:1.0.0

mkdir -p $root_dir
cd $root_dir

# clone repo
git clone https://holoanton@bitbucket.org/kazendi/holoportrt.git
cd $holoportrt

# build
sudo docker build -t $name .

# test
test_cmd=python3 holoport/tests/test_vibe.py holoport/conf/azure/vibe_conf_azure.yaml
sudo docker run --gpus all --rm --ipc=host -it $name $test_cmd
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    lsb-release \
    gnupg2 \
    nano

RUN apt-get install -y python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y python3-pip

RUN mkdir -p /root/dfnet_nerfacto

WORKDIR /root/dfnet_nerfacto

RUN mkdir -p /root/dfnet_nerfacto/nerfstudio && \
    git clone https://github.com/nerfstudio-project/nerfstudio.git /root/dfnet_nerfacto/nerfstudio && \
    cd /root/dfnet_nerfacto/nerfstudio && \
    git checkout tags/v0.3.4

RUN pip3 install --upgrade pip setuptools

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.9

RUN cd /root/dfnet_nerfacto/nerfstudio && pip3 install --ignore-installed -e .

RUN pip3 uninstall torch torchvision -y

RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10-dev

RUN pip3 install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html

RUN TCNN_CUDA_ARCHITECTURES=86 pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

WORKDIR /root/dfnet_nerfacto

COPY nerfstudio /root/dfnet_nerfacto/nerfstudio

COPY pose_regressor ./pose_regressor/

COPY config_dfnet.txt ./

COPY config_dfnetdm.txt ./

COPY *.py ./

COPY workspace ./workspace






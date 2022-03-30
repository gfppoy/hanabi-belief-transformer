FROM nvidia/cuda:10.2-base
MAINTAINER Darius Muglich

RUN apt-get update && apt-get install -y \
    vim wget unzip \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 build-essential

RUN useradd -u <<UID>> --create-home user
USER user
WORKDIR /home/user

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

# Python packages (add new ones to the list here if necessary)
RUN conda install pytorch=1.5.1 torchvision tensorboard==2.0.0 cudatoolkit=10.2 matplotlib -c pytorch
RUN pip install gym

ADD . /home/user/
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
 #&& apt-get install software-properties-common -y && \
RUN apt-get update -y && apt-get install software-properties-common -y &&  \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git \
    libsm6 libxext6 libxrender-dev python3.7 python3-pip
# python3.7 python3.7-pip


# ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate" >> ~/.bashrc && \
#    conda activate

# Install gcc-7
# && apt install -yq software-properties-common \
RUN apt update -qq \
&& apt install -yq software-properties-common \
&& add-apt-repository -y ppa:ubuntu-toolchain-r/test \
&& apt update -qq \
&& apt install -yq g++-8  \
&& apt clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure alias
RUN update-alternatives \
 --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
 --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
 --slave /usr/bin/gcov gcov /usr/bin/gcov-7 \
 --slave /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-7 \
 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 \
 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 \
 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7

# install conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean
RUN conda update -n base -c defaults conda

RUN conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

#RUN sudo ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10.1 /usr/local/cuda-10.1/lib64/libcublas.so.10.1
# create conda environment and install pytorch in it from source
RUN conda init bash && \
 . ~/.bashrc && \
 conda create --name python=3.7 && \
 conda activate && \
 conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing && \
 conda install -c pytorch magma-cuda110 && \
 pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
 #pip install torch torchvision \
 pip install numpy && \
 pip install psutil && \
 conda install -c conda-forge cmake 
#git clone -b v1.6.0 --recursive https://github.com/pytorch/pytorch && \
 #cd pytorch && \
 #export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
 #LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0" python setup.py install

#RUN conda init bash && \
#    . ~/.bashrc && \
#    conda create -n hanabi python=3.7 && \
#    conda activate hanabi && \
#    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
#    pip install numpy && \
#    pip install pip install psutil && \
#    conda install -c conda-forge cmake

RUN pip3 install tensorboardX psutil
# RUN conda install -c conda-forge cmake

RUN echo 'CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}' >> ~/.bashrc
RUN echo 'export CPATH=${CONDA_PREFIX}/include:${CPATH}' >> ~/.bashrc
RUN echo 'export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}' >> ~/.bashrc
RUN echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc

WORKDIR /install

RUN git clone https://github.com/facebookresearch/hanabi.git hanabi &&  \
    cd hanabi && \
    rm -rf hanabi-learning-environment && \
    git clone https://github.com/hengyuan-hu/hanabi-learning-environment.git hanabi-learning-environment && \
    cd hanabi-learning-environment &&  \
    git checkout 273c5ca7f5 && \
    cd .. && \
    cd third_party && \
    rm -rf pybind11 && \
    git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    git checkout a1b71df && \
    cd ../../ && \
    mkdir build && \
    cd build && \   
    cmake .. && \
    make -j`nproc`

#RUN git clone --recursive https://github.com/facebookresearch/hanabi.git && \
#    cd hanabi && \
#    mkdir build && \
#    cd build && \
#    cmake .. && \
#    make -j10

#  git checkout 1f5fc60 && \
# https://github.com/facebookresearch/hanabi.git

RUN mkdir /sad_lib && cp /install/hanabi/build/*.so /sad_lib && cp /install/hanabi/build/rela/*.so /sad_lib
RUN echo 'export PYTHONPATH=/sad_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> ~/.profile

RUN apt-get install -y p7zip-full
WORKDIR /pyhanabi

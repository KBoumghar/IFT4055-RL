# Base container that includes all dependencies but not the actual repo

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.0

FROM nvidia/cudagl${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
    
# NOTE: we don't use TF so might not need some of these
# ========== Tensorflow dependencies ==========
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        # libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        zip \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
#        apt-get install nvinfer-runtime-trt-repo-ubuntu${UBUNTU_VERSION/./}-5.0.2-ga-cuda${CUDA} \
#        && apt-get update \
#        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y
# RUN apt-get install -y python3-dev python3-pip
RUN apt-get update --fix-missing
RUN apt-get install -y wget bzip2 ca-certificates git vim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        premake4 \
        git \
        curl \
        vim \
        ffmpeg \
	    libgl1-mesa-dev \
	    libgl1-mesa-glx \
	    libglew-dev \
	    libosmesa6-dev \
	    libxrender-dev \
	    libsm6 libxext6 \
        unzip \
        patchelf \
        ffmpeg \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        python3-dev python3-pip graphviz \
        freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglew1.6-dev mesa-utils
        
# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
# COPY ./files/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy
        
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y
RUN apt update
RUN apt install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

RUN conda create --name smirl python=3.7 pip
RUN echo "source activate smirl" >> ~/.bashrc
ENV PATH /opt/conda/envs/smirl/bin:$PATH

RUN mkdir /root/playground

# make sure your domain is accepted
# RUN touch /root/.ssh/known_hosts
RUN mkdir /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN ls
WORKDIR /root/playground/
RUN git clone https://github.com/KBoumghar/rlkit
WORKDIR /root/playground/rlkit
RUN git checkout patch-1 #Might need updating if dependancies are too old
RUN git reset --hard origin/patch-1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        swig
RUN pip install -r requirements.txt
RUN pip install ./
 
# WORKDIR /root/playground
# RUN ls -la
# RUN git clone git@github.com:Neo-X/TerrainRLSim.git
# ENV TERRAINRL_PATH /root/playground/TerrainRLSim
# WORKDIR /root/playground/TerrainRLSim
# RUN wget https://github.com/UBCMOCCA/TerrainRLSim/releases/download/0.8/TerrainRLSim_external_June_21_2019.tar.xz
# RUN tar -xvf TerrainRLSim_external_June_21_2019.tar.xz
# RUN apt-get update
# RUN chmod +x ./deb_deps.sh && ./deb_deps.sh
# RUN cd external/caffe && make clean && make
# RUN cp -r external/caffe/build/lib . && cp external/caffe/build/lib/libcaffe.* lib/ && cp external/Bullet/bin/*.so lib/ && cp external/jsoncpp/build/debug/src/lib_json/*.so* lib/
# RUN cd simAdapter/ && apt-get install swig3.0 python3-dev python3-pip -y && chmod +x ./gen_swig.sh && ./gen_swig.sh
# RUN ls -la
# RUN chmod +x ./premake4_linux && ./premake4_linux gmake
# RUN cd gmake && make config=release64 -j 6
# RUN pip install -v -e $TERRAINRL_PATH
# RUN pip install -r requirements.txt
WORKDIR /root/playground

## Install VizDoom dependancies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev liblua5.1-dev

RUN ls

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN conda list
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
#RUN conda install pytorch==1.6.0 torchvision=0.2.0 cudatoolkit=10.2.89 -c pytorch

RUN ls

COPY . /root/playground/SMiRL_Code
WORKDIR /root/playground/SMiRL_Code
RUN pip install -r requirements.txt
RUN pip install ./

WORKDIR /root/playground/
RUN git clone https://github.com/Neo-X/doodad/

WORKDIR /root/playground/doodad/
RUN git checkout surprise
RUN git reset --hard origin/surprise

RUN export PYTHONPATH=$PYTHONPATH:/root/playground/doodad
RUN pip install -r requirements.txt
RUN pip install ./
WORKDIR /root/playground/SMiRL_Code

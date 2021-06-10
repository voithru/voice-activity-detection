FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

WORKDIR /workspace

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    wget \
    libsndfile1 \
    libgl1-mesa-glx

# Install python
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O Miniconda.sh && \
    bash Miniconda.sh -b -p /root/miniconda3 && \
    rm Miniconda.sh
ENV PATH /root/miniconda3/bin:$PATH

# For opencv-python
RUN conda install -c conda-forge opencv -y

# Cache heavy libraries first
RUN pip install --no-cache-dir numpy==1.16.5
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

COPY . .

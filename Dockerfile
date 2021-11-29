FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get -y full-upgrade && \
    apt-get install -y \
        curl wget unzip git \
        libmpich-dev libjpeg8-dev zlib1g-dev libtiff5-dev python3-pip \
        ffmpeg libsm6 libxext6

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR text_generation_bert

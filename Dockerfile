FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# time zone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install basic dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    sudo git wget cmake nano vim gcc g++ build-essential ca-certificates software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install python
RUN add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y python3.7 \
&& apt-get update \
&& apt-get install -y  python3.7-distutils \
&& wget -O ./get-pip.py https://bootstrap.pypa.io/get-pip.py \
&& python3.7 ./get-pip.py \
&& ln -s /usr/bin/python3.7 /usr/local/bin/python3 \
&& ln -s /usr/bin/python3.7 /usr/local/bin/python

# install common python packages
RUN apt-get update \
&& apt-get install -y libgl1-mesa-dev
ADD ./requirements.txt /tmp
RUN pip install pip setuptools -U && pip install -r /tmp/requirements.txt

#zip,unzipをインストール
RUN apt-get update \
&& apt-get install -y zip \
&& apt-get install -y unzip

# コード補完用ライブラリをインストール
RUN pip install jupyter-contrib-nbextensions
RUN pip install jupyter-nbextensions-configurator

# 拡張機能を有効化する
RUN jupyter contrib nbextension install
RUN jupyter nbextensions_configurator enable

# set working directory
WORKDIR /root/user

# config and clean up
RUN ldconfig \
&& apt-get clean \
&& apt-get autoremove
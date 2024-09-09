FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/root

# Basic Ubuntu stuff
RUN chmod 777 /tmp
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && apt-get update && apt-get -y upgrade
RUN apt-get install -y net-tools iputils-ping git build-essential cmake pkg-config
RUN apt install -y nano
# Python
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt update && apt upgrade
RUN apt-get install -y python3.10 
RUN apt-get install -y pip python3.10-dev python3.10-venv
RUN python3.10 -m venv /root/venv && echo "source /root/venv/bin/activate" >> /root/.bashrc
ENV PATH="/root/venv/bin:$PATH"

RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk2.0-dev libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev



RUN pip install --upgrade pip
RUN pip install --no-cache-dir open3d

RUN pip install --no-cache-dir numpy==1.26.4

# RUN apt-get install -y python3-opencv nano
RUN pip install --no-cache-dir opencv-contrib-python

# Astra python

# Realsense python
RUN pip install --no-cache-dir pyrealsense2

RUN apt-get install -y cmake gcc g++

RUN git clone https://github.com/opencv/opencv.git

RUN cd opencv && mkdir build && cd build && cmake ../ -DOPENCV_GENERATE_PKGCONFIG=ON

RUN cd /opencv/build && make && make install

RUN ldconfig


# Define working directory.
WORKDIR /root
# Define default command.
CMD ["bash"]
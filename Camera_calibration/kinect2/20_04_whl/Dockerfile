FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /root

# Define working directory.
WORKDIR /root

RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && apt-get update && apt-get -y upgrade && apt-get install -y python3 && apt-get install -y pip && pip install --upgrade pip

RUN apt-get install -y git && git clone https://github.com/OpenKinect/libfreenect2.git

RUN apt-get install -y build-essential cmake pkg-config libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev

RUN cd libfreenect2 && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 && make && make install

# RUN git clone https://github.com/r9y9/pylibfreenect2 && cd pylibfreenect2 && git checkout v0.1.4 && 
RUN pip install --no-cache-dir numpy && pip install --no-cache-dir cython==0.29.37 && mkdir whl && LIBFREENECT2_INSTALL_PREFIX=~/freenect2 pip install --cache-dir="~/whl" pylibfreenect2

# RUN apt-get install -y python3-opencv

RUN echo "export LD_LIBRARY_PATH=~/freenect2/lib" >> /root/.bashrc
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /root

# Define working directory.
WORKDIR /root

# Basic Ubuntu stuff
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && apt-get update && apt-get -y upgrade
RUN apt-get install -y software-properties-common  net-tools iputils-ping netcat git build-essential cmake pkg-config

# Python
RUN apt-get install -y python3 pip python-dev && pip install --upgrade pip

# Camera's dependencies
RUN apt-get install -y libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev libxmu-dev libxi-dev freeglut3-dev


# Kinect base software
RUN git clone https://github.com/OpenKinect/libfreenect/ cd libfreenect && git checkout v0.7.5 && mkdir build && cd build && cmake .. -DBUILD_PYTHON3=ON && make && make install && ldconfig /usr/local/lib64/

# Kinect 2 base software
RUN git clone https://github.com/OpenKinect/libfreenect2.git && cd libfreenect2 && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 && make && make install && echo "export LD_LIBRARY_PATH=~/freenect2/lib" >> /root/.bashrc

# Astra
# ...............

# Basler
RUN apt-get install -y pylon

# Realsense


# Kinect Python

# Kinect2 Python
COPY ./pylibfreenect2-0.1.4-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/pylibfreenect2-0.1.4-cp310-cp310-linux_x86_64.whl

# Basler Python
RUN python3 -m pip install pypylon

RUN apt-get install -y python3-opencv

RUN 

# Define default command.
CMD ["bash"]
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/root

# Basic Ubuntu stuff
RUN chmod 777 /tmp
RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && apt-get update && apt-get -y upgrade
RUN apt-get install -y net-tools iputils-ping netcat git build-essential cmake pkg-config
RUN apt install -y nano
# Python
RUN apt-get install -y python3 pip python3-dev python3-venv && pip install --upgrade pip
RUN python3 -m venv /root/venv && echo "source /root/venv/bin/activate" >> /root/.bashrc
ENV PATH="/root/venv/bin:$PATH"

# Camera's dependencies
RUN apt-get install -y libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev libxmu-dev libxi-dev freeglut3-dev


RUN pip install --upgrade pip
RUN pip install --no-cache-dir open3d


RUN pip install --no-cache-dir numpy==1.24.4 && pip install --no-cache-dir cython==3.0.10

# Kinect base software
RUN git clone https://github.com/OpenKinect/libfreenect/ && cd libfreenect && git checkout v0.7.5 && mkdir build && cd build && cmake .. -DBUILD_PYTHON3=ON && make && make install && ldconfig /usr/local/lib64/

# Kinect 2 base software
RUN git clone https://github.com/OpenKinect/libfreenect2.git && cd libfreenect2 && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 && make && make install && echo "export LD_LIBRARY_PATH=~/freenect2/lib" >> /root/.bashrc

# Remove specific versions 
RUN pip uninstall -y numpy && pip uninstall -y cython

RUN pip install --no-cache-dir numpy==1.26.4

# RUN apt-get install -y python3-opencv nano
RUN pip install --no-cache-dir opencv-contrib-python

# Astra
# ...............

# Basler
# RUN apt install -y wget && wget https://www2.baslerweb.com/media/downloads/software/pylon_software/pylon-7.5.0.15658-linux-x86_64_debs.tar.gz
# RUN tar -xvf pylon-7.5.0.15658-linux-x86_64_debs.tar.gz && apt install ./pylon_7.5.0.15658-deb0_amd64.deb
COPY ./* /tmp/
RUN apt install /tmp/pylon_7.5.0.15658-deb0_amd64.deb

# Realsense
# ................
# Kinect Python
# COPY ./freenect-0.0.0-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/freenect-0.0.0-cp310-cp310-linux_x86_64.whl

# Kinect2 Python
# COPY ./pylibfreenect2-0.1.4-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/pylibfreenect2-0.1.4-cp310-cp310-linux_x86_64.whl

# Basler Python
RUN python3 -m pip install --no-cache-dir pypylon

# Astra python

# Realsense python
RUN pip install --no-cache-dir pyrealsense2

RUN apt install libopencv-dev


# Define working directory.
WORKDIR /root
# Define default command.
CMD ["bash"]
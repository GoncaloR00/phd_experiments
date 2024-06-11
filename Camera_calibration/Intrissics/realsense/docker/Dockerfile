FROM osrf/ros:noetic-desktop-full

ARG DEBIAN_FRONTEND=noninteractive

RUN \
sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
apt-get update && \
apt-get -y upgrade && \
echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
echo "source ~/catkin_ws/devel/setup.bash" >> /root/.bashrc &&\
apt-get install -y build-essential && \
apt-get install -y net-tools && \
apt-get install -y iputils-ping && \
apt-get install -y netcat
RUN apt install python3 && \
apt-get install -y pip &&\
pip install --no-cache-dir numpy && \
pip install --no-cache-dir pyrealsense2 && \
apt-get install -y terminator && \
apt-get install -y libuvc-dev && \
apt-get install -y ros-noetic-rgbd-launch ros-noetic-libuvc-camera ros-noetic-libuvc-ros && \
# rosdep install camera_calibration && \
apt-get install -y ros-noetic-camera-calibration
RUN mkdir -p ~/catkin_ws/src && \
cd ~/catkin_ws/ && \
/bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

ENV HOME /root


# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]
# Pull base image.
FROM ubuntu:20.04

# Install.
ARG DEBIAN_FRONTEND=noninteractive

RUN \
  # Basic ubuntu stuff
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y net-tools && \
  apt-get install -y iputils-ping && \
  apt-get install -y netcat && \
  apt-get install -y vim && \
  # Terminator
  apt-get install -y terminator && \
  # Python
  apt install python3 && \
  apt-get install -y pip &&\
  pip install --no-cache-dir numpy && \
  # ROS
  sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
  apt-get install -y curl && \
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
  apt-get update && \
  apt-get install -y ros-noetic-desktop-full && \
  echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
  echo "source ~/catkin_ws/devel/setup.bash" >> /root/.bashrc &&\
  apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
  apt-get install -y python3-opencv && \
  apt-get install -y ros-noetic-cv-bridge && \
  apt-get install -y ros-noetic-ros-numpy qt5-default && \
  apt-get install -y ros-noetic-rviz-visual-tools && \
  apt-get install -y qtbase5-dev && \
  apt-get install -y qtdeclarative5-dev && \
  # ROS catkin workspace
  apt-get -y install ros-noetic-catkin && \
  mkdir -p ~/catkin_ws/src && \
  cd ~/catkin_ws/ && \
  /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

RUN cd ~/catkin_ws/src && \
    git clone https://github.com/lardemua/atom && \
    cd atom && \
    pip3 install -r requirements.txt && \
    mkdir -p ~/bagfiles && \
    mkdir -p ~/datasets && \
    mkdir -p ~/catkin_ws/src/external && \
    echo "alias cc='cd ~/catkin_ws && catkin_make && cd -'" >> /root/.bashrc && \
    echo "PS1='\[\e[36m\]\w \[\e[34m\]ATOM DOCKER -->\[\e[00m\] '" >> /root/.bashrc && \
    echo "export ROS_BAGS='$HOME/bagfiles'" >> /root/.bashrc && \
    echo "export ATOM_DATASETS='$HOME/datasets'" >> /root/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/catkin_ws/src/atom/atom_worlds/pattern/models" >> /root/.bashrc && \
    cd ~/catkin_ws/ && /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]

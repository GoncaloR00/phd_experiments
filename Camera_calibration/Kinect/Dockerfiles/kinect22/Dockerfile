# Pull base image.
FROM ubuntu:22.04

# Install.
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

RUN \
  # Basic ubuntu stuff
  chmod 777 /tmp &&\
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y net-tools && \
  apt-get install -y iputils-ping && \
  apt-get install -y netcat && \
  # Python
  apt install python3 && \
  apt-get install -y pip && pip install --upgrade pip &&\
  # apt-get install -y python-dev && \
  pip install numpy==1.24.4 && \
  pip install cython==3.0.10 && \
  apt-get install -y git cmake freeglut3-dev pkg-config libxmu-dev libxi-dev libusb-1.0-0-dev && \
  git clone https://github.com/OpenKinect/libfreenect/ && \
  cd libfreenect && \
  git checkout v0.7.5 && \
  mkdir build && \
  cd build && \
  cmake .. -DBUILD_PYTHON3=ON && \
  make && \
  make install && \
  ldconfig /usr/local/lib64/
RUN cd libfreenect/wrappers/python/ && sed -i 's/version = Cython.Compiler.Main.Version.version/version = "3.0.10"/' setup.py
  # pip install wheel && \
  # python3 setup.py bdist_wheel



# Define default command.
CMD ["bash"]

# Pull base image.
FROM ubuntu:22.04

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
  apt-get install -y python3-opencv

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]


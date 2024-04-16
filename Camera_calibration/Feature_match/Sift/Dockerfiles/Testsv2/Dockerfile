# Pull base image.
FROM teste1

# Install.
ARG DEBIAN_FRONTEND=noninteractive

RUN \
  # Basic ubuntu stuff
  apt-get update && \
  apt-get -y upgrade && \
  apt-get -y install git && \
  apt-get -y install libcanberra-gtk-module libcanberra-gtk3-module


# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]


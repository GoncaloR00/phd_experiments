#!/bin/bash
docker run -it --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/dri:/dev/dri \
    -v /dev:/dev \
    --ulimit nofile=1024:524288 \
    --ipc=host \
    --network host \
    --privileged\
    --name freenect --rm \
    libfreenect
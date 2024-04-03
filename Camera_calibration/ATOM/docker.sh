#!/bin/bash
files_dir="$(cd "$(dirname "$0")" && pwd)/Files"
docker run -it --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/dri:/dev/dri \
    --ulimit nofile=1024:524288 \
    --ipc=host \
    --network host \
    --name atom_container --rm \
    -v "$files_dir/src/":/root/catkin_ws/src/external \
    -v "$files_dir/bagfiles/":/root/bagfiles \
    -v "$files_dir/datasets/":/root/datasets \
    atom terminator

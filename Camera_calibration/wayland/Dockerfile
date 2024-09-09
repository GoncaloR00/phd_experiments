FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /home

# Define working directory.
WORKDIR /home

RUN sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && apt-get update && apt-get -y upgrade

RUN apt-get update && apt-get -y upgrade && apt-get install -y terminator
RUN apt-get install -y tofi
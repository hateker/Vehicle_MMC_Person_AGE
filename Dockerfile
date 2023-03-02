FROM ubuntu:20.04

RUN apt update -y

RUN apt install sudo software-properties-common libglu1-mesa-dev -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install python3.8 python3-pip -y

RUN pip install -U pip 

RUN pip install uvicorn==0.20

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu

RUN echo 'ubuntu:12345' | chpasswd

USER ubuntu

WORKDIR /home/ubuntu

COPY --chown=ubuntu:root Docker_Folder /home/ubuntu/Docker_Folder

RUN cd /home/ubuntu/Docker_Folder

RUN pip install -U pip 

RUN pip install -r /home/ubuntu/Docker_Folder/requirement.txt

RUN chmod +x /home/ubuntu/Docker_Folder/main.sh

ENTRYPOINT /home/ubuntu/Docker_Folder/main.sh
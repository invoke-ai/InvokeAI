FROM nvidia/cuda:11.7.1-base-ubuntu22.04
FROM continuumio/miniconda3

RUN apt update
RUN apt install dos2unix -y

WORKDIR /
COPY . /InvokeAI/

WORKDIR /InvokeAI

RUN mkdir -p models/ldm/stable-diffusion-v1
#RUN pwd
#RUN ls -la
RUN mv model.ckpt models/ldm/stable-diffusion-v1/model.ckpt

RUN conda env create -f environment.yml
RUN conda env list
SHELL ["conda", "run", "-n", "invokeai", "/bin/bash", "-c"]

RUN python3 scripts/preload_models.py

#WORKDIR /
#ADD /docker-build/start.sh .
RUN dos2unix docker-build/entrypoint.sh
RUN chmod +x docker-build/entrypoint.sh

ENTRYPOINT docker-build/entrypoint.sh

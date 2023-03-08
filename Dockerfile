FROM nvidia/cuda:11.7.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip git unzip ffmpeg libsm6 libxext6 vim

RUN pip install --upgrade pip

RUN pip install wandb opencv-python gdown

RUN pip install -qq git+https://github.com/jasdeep06/diffusers

RUN pip install -q -U --pre triton

RUN pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

RUN pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy

RUN pip install -U xformers

RUN pip install torchvision
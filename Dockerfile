FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y wget curl git build-essential
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && apt-get install -y wget curl git build-essential

# Miniconda
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 

# RUN conda create -n xp
# SHELL ["conda", "run", "--no-capture-output", "-n", "xp", "/bin/bash", "-c"]
# RUN conda install python=3.7 \
#     && conda install -c conda-forge jupyterlab \
#     && conda init bash \
#     && echo "conda activate xp" >> ~/.bashrc

RUN pip install ipykernel jupyterlab jupyter_http_over_ws \
    && jupyter serverextension enable --py jupyter_http_over_ws

WORKDIR /content/
COPY . /content/DECA/
#build DECA
# docker build -t deca:latest .
#run container with jupyter lab
#docker run -it --rm -p 8888:8888 -v $(pwd):/content/DECA/ --gpus all --ipc=host --name deca deca:latest jupyter lab
#run container with compose
#docker-compose up -d
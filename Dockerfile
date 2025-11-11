# Multi-GPU development image for ColonSuperpoinTorch
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CONDA_DIR=/opt/conda \
    CONDA_ENV=py38-sp \
    PATH=/opt/conda/bin:$PATH

# Base system packages required for building Python wheels and OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and create project environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniconda.sh && \
    "$CONDA_DIR/bin/conda" update -y -n base conda && \
    "$CONDA_DIR/bin/conda" clean -afy

SHELL ["/bin/bash", "-lc"]

RUN conda create -y -n "$CONDA_ENV" python=3.8 pip && \
    conda clean -afy

# Install PyTorch with CUDA 11.3 binaries plus project Python dependencies
WORKDIR /workspace/ColonSuperpoinTorch
COPY requirements_py38.txt requirements_py38.txt

RUN . "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda activate "$CONDA_ENV" && \
    conda install -y -c pytorch pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 && \
    pip install --no-cache-dir -r requirements_py38.txt && \
    conda clean -afy

# Copy repository contents
COPY . .

ENV PATH="$CONDA_DIR/envs/$CONDA_ENV/bin:$CONDA_DIR/bin:$PATH" \
    CONDA_DEFAULT_ENV="$CONDA_ENV"

CMD ["bash"]

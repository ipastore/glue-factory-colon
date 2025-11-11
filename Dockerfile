ARG CUDA_TAG=11.4.3-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_TAG}

LABEL org.opencontainers.image.source="https://github.com/cvg/glue-factory"
LABEL org.opencontainers.image.description="Glue Factory development image with CUDA 11.4 runtime, PyTorch, and optional extras."

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# ENV HOME=/home/hostuser \
#     XDG_CACHE_HOME=/home/hostuser/.cache \
#     MPLCONFIGDIR=/home/hostuser/.config/matplotlib \
#     TORCH_HOME=/home/hostuser/.cache/torch


SHELL ["/bin/bash", "-lc"]

# System dependencies for building CUDA software, Ceres, and Python wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        gnupg \
        libatlas-base-dev \
        libboost-all-dev \
        libc6-dev \
        libeigen3-dev \
        libffi-dev \
        libgflags-dev \
        libglib2.0-0 \
        libglu1-mesa \
        libgoogle-glog-dev \
        liblapack-dev \
        libmetis-dev \
        libopenblas-dev \
        libsm6 \
        libsuitesparse-dev \
        libx11-6 \
        libxext6 \
        libxi6 \
        libxmu6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        unzip \
        wget && \
    rm -rf /var/lib/apt/lists/*

ARG MAMBAFORGE_VERSION=24.5.0-0
ARG CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/Mambaforge-Linux-x86_64.sh -o /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/mambaforge.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:${PATH}

ARG PYTHON_VERSION=3.10
RUN ${CONDA_DIR}/bin/mamba create -y -n gluefactory python=${PYTHON_VERSION} pip && \
    ${CONDA_DIR}/bin/conda clean -afy

ENV CONDA_DEFAULT_ENV=gluefactory \
    PATH=${CONDA_DIR}/envs/gluefactory/bin:${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace
COPY . /workspace

ARG HOST_UID=1001
ARG HOST_GID=1001
RUN groupadd -g ${HOST_GID} hostuser || true && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash hostuser || true && \
    mkdir -p /home/hostuser && \
    chown -R ${HOST_UID}:${HOST_GID} /home/hostuser /workspace

# RUN mkdir -p /home/hostuser/.cache /home/hostuser/.config/matplotlib /home/hostuser/.cache/torch && \
#     chown -R ${HOST_UID}:${HOST_GID} /home/hostuser/.cache /home/hostuser/.config/matplotlib /home/hostuser/.cache/torch

# Pre-install GPU-enabled PyTorch compatible with both target drivers
ARG PYTORCH_VERSION=1.12.1
ARG TORCHVISION_VERSION=0.13.1
ARG PYTORCH_CUDA_TAG=cu113
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu113
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate gluefactory && \
    python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir --extra-index-url ${PYTORCH_INDEX_URL} \
        torch==${PYTORCH_VERSION}+${PYTORCH_CUDA_TAG} \
        torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA_TAG}

# Install Glue Factory with extras + developer tooling
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate gluefactory && \
    python -m pip install --no-cache-dir pybind11 && \
    python -m pip install --no-cache-dir -e '.[dev]' && \
    python -m pip install --no-cache-dir \
        pycolmap \
        poselib \
        git+https://github.com/iago-suarez/pytlsd.git@4180ab8990ae68cc9c8797c63aa1dc47b2c714da \
        git+https://github.com/cvg/DeepLSD.git && \
    python -m pip install --no-cache-dir tornado pytest pytest-cov

USER hostuser
CMD ["/bin/bash"]

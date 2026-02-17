# Multi-GPU development image for Glue Factory with CUDA-enabled Pycolmap
# plus CudaSift Python wrapper (cudasift_py).
ARG CUDA_IMAGE=nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ARG CONDA_DIR=/opt/conda
ARG CONDA_ENV=gluefactory
ARG PYTHON_VERSION=3.10
ARG CUDATOOLKIT_VERSION=11.4
ARG PYTORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG CUDASIFT_REPO=https://github.com/ipastore/CudaSift-py-wrapper.git
ARG CUDASIFT_CUDA_ARCHS=70
ARG CUDASIFT_DIR=/opt/CudaSift-py-wrapper
ARG USERNAME=dev
ARG USER_UID=1000

ENV CONDA_DIR=${CONDA_DIR} \
    CONDA_ENV=${CONDA_ENV} \
    PATH=${CONDA_DIR}/bin:$PATH \
    CONDA_OVERRIDE_CUDA=${CUDATOOLKIT_VERSION}

# Base system packages required for building Python wheels, OpenCV runtime, and CudaSift wrapper.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libceres-dev \
        libeigen3-dev \
        libopencv-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and create project environment.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    "${CONDA_DIR}/bin/conda" update -y -n base conda && \
    "${CONDA_DIR}/bin/conda" clean -afy

SHELL ["/bin/bash", "-lc"]

RUN conda create -y -n "${CONDA_ENV}" python=${PYTHON_VERSION} pip && \
    conda clean -afy

# Install PyTorch, CUDA toolkit, CUDA-enabled Pycolmap, and pybind11 from conda-forge.
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    conda install -y --strict-channel-priority -c conda-forge \
        cudatoolkit=${CUDATOOLKIT_VERSION} \
        nccl=2.14.3 \
        pytorch=${PYTORCH_VERSION} \
        torchvision=${TORCHVISION_VERSION} \
        pybind11 \
        'pycolmap>=0.4.0' && \
    conda clean -afy

# Ensure pycolmap exposes CUDA support at runtime.
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python - <<'PYCODE'
import pycolmap
import torch

print(f"pycolmap version: {pycolmap.__version__}")
print(f"pycolmap CUDA available: {pycolmap.has_cuda}")
assert pycolmap.has_cuda, "pycolmap was built without CUDA support"
print(f"torch version: {torch.__version__}")
print(f"torch.cuda available: {torch.cuda.is_available()}")
print(f"CUDA devices visible to torch: {torch.cuda.device_count()}")
PYCODE

WORKDIR /workspace

# Copy repository contents.
COPY . .

# Install Glue Factory and optional extras (except pycolmap which comes from conda).
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --no-deps -e . && \
    python -m pip install --no-cache-dir \
        numpy \
        opencv-python \
        tqdm \
        matplotlib \
        scipy \
        h5py \
        omegaconf \
        tensorboard \
        albumentations \
        seaborn \
        joblib \
        "scikit-learn==1.3.2" \
        "kornia==0.6.12" && \
    python -m pip install --no-cache-dir --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git" && \
    python -m pip install --no-cache-dir poselib

# Build CudaSift Python wrapper in release mode and validate import.
# Keep it outside /workspace so bind-mounting /workspace at runtime does not hide it.
RUN mkdir -p /opt && \
    git clone "${CUDASIFT_REPO}" "${CUDASIFT_DIR}" && \
    . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    cd "${CUDASIFT_DIR}" && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDASIFT_CUDA_ARCHS}" .. && \
    cmake --build . --target cudasift_py -- -j1 && \
    python -c "import cudasift_py; print(cudasift_py.__file__)"

# Durable path to cudasift_py for every shell/session.
ENV PYTHONPATH="${CUDASIFT_DIR}/build"

# Create default non-root user (configurable via build args).
RUN useradd -m -u "${USER_UID}" -s /bin/bash "${USERNAME}" && \
    chmod 755 "/home/${USERNAME}" && \
    chown -R "${USERNAME}:${USERNAME}" /workspace

ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:$PATH" \
    CONDA_DEFAULT_ENV="${CONDA_ENV}"

USER ${USERNAME}

CMD ["bash"]

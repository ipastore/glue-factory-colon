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
ARG CUDASIFT_CUDA_ARCHS_SM=7.0
ARG CUDASIFT_DIR=/opt/CudaSift-py-wrapper
ARG ROMA_REPO=https://github.com/Parskatt/RoMa.git
ARG ROMA_DIR=/opt/RoMa
ARG USERNAME=dev
ARG USER_UID=1000
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu114

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

# Install CUDA-enabled Pycolmap and pybind11 from conda-forge.
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    conda install -y -c conda-forge \
        pybind11 \
        ncurses \
        'pycolmap==0.4.0' && \
    conda clean -afy

# Install PyTorch and torchvision via pip from the matching CUDA wheel index.
# This avoids conda/pip conflicts by keeping PyTorch separate from conda-managed packages.
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --no-cache-dir \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url ${PYTORCH_INDEX_URL}
        
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

# ── 1. Pure-Python dependencies (order-independent) ──────────────────────
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip && \
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
        einops \
        loguru \
        timm \
        wandb \
        "poselib>=2.0.4" \
        "scikit-learn==1.3.2" \
        "kornia>=0.6.12"

# ── 2. CUDA C++ extension – build from source against the installed torch ─
#    --no-build-isolation → sees the already-installed torch headers
#    --no-deps            → don't let pip pull a different torch
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    TORCH_CUDA_ARCH_LIST="${CUDASIFT_CUDA_ARCHS_SM}" \
    python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        "fused-local-corr @ git+https://github.com/Parskatt/fused-local-corr.git"

# ── 3. Git-only packages (--no-deps, all their deps already installed) ────
#    LightGlue before RoMa and GlueFactory since GF lists it as a dep.
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --no-cache-dir --no-deps \
        "lightglue @ git+https://github.com/cvg/LightGlue.git" && \
    python -m pip install --no-cache-dir --no-deps \
        "romatch @ git+https://github.com/Parskatt/RoMa.git"

# ── 4. Glue Factory itself (editable, --no-deps) ─────────────────────────
WORKDIR /workspace
COPY . .
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --no-cache-dir --no-deps -e .

# ── 5. Validation ────────────────────────────────────────────────────────
RUN . "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_ENV}" && \
    python - <<'PYCODE'
import pycolmap, torch, local_corr
from romatch.models.model_zoo.roma_models import roma_model

print(f"torch version:    {torch.__version__}")
print(f"torch CUDA:       {torch.cuda.is_available()}")
print(f"pycolmap version: {pycolmap.__version__}")
print(f"pycolmap CUDA:    {pycolmap.has_cuda}")
print(f"local_corr:       {local_corr.__file__}")
print(f"RoMa:             {roma_model.__name__}")
assert pycolmap.has_cuda, "pycolmap built without CUDA"
PYCODE


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
    CONDA_DEFAULT_ENV="${CONDA_ENV}" \
    LD_LIBRARY_PATH="${CONDA_DIR}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

USER ${USERNAME}

CMD ["bash"]

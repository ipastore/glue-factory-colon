# # 5090 - ecs - official
# docker build -t official-gluefactory:cuda12.8 \
# --build-arg USERNAME=ecs \
# --build-arg USER_UID=1001 \
# --build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 \
# --build-arg CUDATOOLKIT_VERSION=12.8 \
# --build-arg PYTORCH_VERSION=2.7.0 \
# --build-arg TORCHVISION_VERSION=0.22.0 \
# --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
# --build-arg CUDASIFT_CUDA_ARCHS=120 \
# . 2>&1 | tee ./docker_build.log
# # --no-cache

### NEED TO UPDATE THIS IMAGE TO USE A NEWER VERSION OF TORCH greater or equal than 2.5.1. MAYBE we could use the same pytorch version and cuda for everybody
# # 4090 - server - official 
# docker build -t official-gluefactory:cuda11.8 \
# --build-arg USERNAME=server \
# --build-arg USER_UID=1000 \
# --build-arg CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 \
# --build-arg CUDATOOLKIT_VERSION=11.8 \
# --build-arg PYTORCH_VERSION=2.0.1 \           
# --build-arg TORCHVISION_VERSION=0.15.2 \
# --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
# --build-arg CUDASIFT_CUDA_ARCHS=89 \
# . 2>&1 | tee ./docker_build.log
# # --no-cache

# # 2080 - student - gluefactory-colon
# docker build -t gluefactory:cuda12.8 \
#     --build-arg USERNAME=student \
#     --build-arg USER_UID=1001 \
#     --build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 \
#     --build-arg CUDATOOLKIT_VERSION=12.8 \
#     --build-arg PYTORCH_VERSION=2.7.0 \
#     --build-arg TORCHVISION_VERSION=0.22.0 \
#     --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     --build-arg CUDASIFT_CUDA_ARCHS=75 \
#     --no-cache \
#     . 2>&1 | tee ./docker_build.log


# SAFEST ALTERNATIVE FOR ALL MACHINES
docker build -t gluefactory:cuda11.8 \
    --build-arg USERNAME=student \
    --build-arg USER_UID=1001 \
    --build-arg CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
    --build-arg CUDATOOLKIT_VERSION=11.8 \
    --build-arg PYTORCH_VERSION=2.5.1 \
    --build-arg TORCHVISION_VERSION=0.20.1 \
    --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
    --build-arg CUDASIFT_CUDA_ARCHS=75 \
    --build-arg CUDASIFT_CUDA_ARCHS_SM=7.5 \
    --no-cache \
    . 2>&1 | tee ./docker_build.log


# # DGX - ipastore - gluefactory-colon
# docker build -t gluefactory:cuda11.4 \
# --build-arg USERNAME=ipastore \
# --build-arg USER_UID=18128 \
# --build-arg CUDA_IMAGE=nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 \
# --build-arg CUDATOOLKIT_VERSION=11.4 \
# --build-arg PYTORCH_VERSION=1.13.1 \
# --build-arg TORCHVISION_VERSION=0.14.1 \
# --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu114 \
# --build-arg CUDASIFT_CUDA_ARCHS=70 \
# . 2>&1 | tee ./docker_build.log
# # --no-cache
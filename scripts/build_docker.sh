docker build -t gluefactory:cuda11.4 \
  --build-arg USERNAME=student \
  --build-arg USER_UID=1001 \
  --build-arg CUDASIFT_CUDA_ARCHS=75 \
  --no-cache \
  .

# docker build -t gluefactory:cuda11.8 \
#   --build-arg USERNAME=server \
#   --build-arg USER_UID=1000 \
#   --build-arg CUDASIFT_CUDA_ARCHS=89 \
#   .
  # --no-cache .  2>&1 | tee ./docker_build.log
#!/bin/bash

#SBATCH -J train_sift_lg_ednomapper
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/dev/null
  
# per-run log dir (group by job name)
LOGROOT="${SLURM_SUBMIT_DIR}/dgx_logs/${SLURM_JOB_NAME}"
mkdir -p "$LOGROOT"

CONFIG_FILE="sift+lightglue_megadepth.yaml"

CONFIG_ROOT="gluefactory/configs/"
CONFIG_PATH="${CONFIG_ROOT}${CONFIG_FILE}"

srun --container-mounts=/home/ropert/ipastore/storage/glue-factory-colon:/workspace \
--container-workdir=/workspace \
--container-image=./gluefactory.sqsh \
--output="${LOGROOT}/%j_%t.out" \
--error="${LOGROOT}/%j_%t.err" \
/bin/bash -lc "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64; unset LD_PRELOAD NCCL_LIBRARY; export NCCL_DEBUG=${NCCL_DEBUG:-INFO}; export GLUEFACTORY_DEBUG_CUDA_NCCL=${GLUEFACTORY_DEBUG_CUDA_NCCL:-1}; if [ \"${RUN_NCCL_PRECHECK:-1}\" = \"1\" ]; then conda run -n gluefactory python debug_CUDA_NCCL_when_distributed.py; fi; conda run -n gluefactory python -m gluefactory.train ${SLURM_JOB_NAME} --conf ${CONFIG_PATH} --distributed"


# /bin/bash -lc "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64; unset LD_PRELOAD NCCL_LIBRARY; conda run -n gluefactory python debug_CUDA_NCCL_when_distributed.py"
# /bin/bash -lc "conda run -n gluefactory python -m gluefactory.train ${SLURM_JOB_NAME} --conf gluefactory/configs/sift+lightglue_megadepth.yaml --distributed"

# /bin/bash -lc "conda run -n gluefactory python -m gluefactory.train ${SLURM_JOB_NAME} --conf gluefactory/configs/CudaSift+lightglue_endomapper.yaml --distributed"

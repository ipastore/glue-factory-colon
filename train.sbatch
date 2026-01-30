#!/bin/bash

#SBATCH -J train_sift_lg_ednomapper
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
  
# per-run log dir (group by job name)
LOGROOT="${SLURM_SUBMIT_DIR}/dgx_logs/${SLURM_JOB_NAME}"
mkdir -p "$LOGROOT"
 
srun --container-mounts=/home/ropert/ipastore/storage/glue-factory-colon:/workspace \
--container-workdir=/workspace \
--container-image=./gluefactory.sqsh \
--output="${LOGROOT}/%j_%t.out" \
--error="${LOGROOT}/%j_%t.err" \
/bin/bash -lc "conda run -n gluefactory python -m gluefactory.train ${SLURM_JOB_NAME}  --conf gluefactory/configs/CudaSift+lightglue_endomapper.yaml"

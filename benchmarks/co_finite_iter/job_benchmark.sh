#!/bin/sh
#SBATCH -J symwan-co
#SBATCH -p 64core
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH -w node03
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -eu

: "${SRC_DIR:?Set SRC_DIR to the symWannier repository root to benchmark}"
: "${TAG:?Set TAG to the benchmark label}"

cd "$SLURM_SUBMIT_DIR"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/user2/minkyu/anaconda3/lib"
: "${BLAS_THREADS:=8}"
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export MKL_DYNAMIC=FALSE
export PYTHONPATH="$SRC_DIR/src"

echo "TAG=$TAG"
echo "SRC_DIR=$SRC_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "BLAS_THREADS=$BLAS_THREADS"
echo "PYTHONPATH=$PYTHONPATH"
echo "Started at $(date)"

/usr/bin/time -p -o "time.${TAG}" \
    python -m symwannier.wannierize -S wannier \
    > "stdout.${TAG}" \
    2> "stderr.${TAG}"

echo "Finished at $(date)"

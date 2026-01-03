#!/bin/bash
# install_physics.sh - Physics Environment Setup (Ubuntu 22.04 | Root)
# Usage: ./install_physics.sh --cpu  OR  ./install_physics.sh --gpu
set -e

# --- 0. Parse Arguments ---
MODE="cpu"
for arg in "$@"; do
  case $arg in
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    *) echo "Unknown option: $arg. Use --cpu or --gpu"; exit 1 ;;
  esac
done

echo "===================================================="
echo "INSTALLATION STARTING: MODE=${MODE^^}"
echo "===================================================="

# --- 1. System Dependencies ---
echo "--- 1. Installing System Compilers & MPI ---"
apt update && apt install -y \
    python3-pip python3-dev build-essential gfortran \
    libopenblas-dev libopenmpi-dev openmpi-bin git

# --- 2. Environment Configuration ---
echo "--- 2. Configuring Shell & Hardware Paths ---"
# Find CUDA path automatically for GPU mode
CUDA_PATH=$(find /usr/local -name "cuda-*" -type d -print -quit 2>/dev/null || echo "/usr/local/cuda")

{
    echo "export OMPI_ALLOW_RUN_AS_ROOT=1"
    echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    echo "export OMP_NUM_THREADS=4"
    echo 'export PATH="$HOME/.local/bin:'"$CUDA_PATH"'/bin:$PATH"'
    echo 'export LD_LIBRARY_PATH="'"$CUDA_PATH"'/lib64:$LD_LIBRARY_PATH"'
    if [ "$MODE" = "gpu" ]; then
        echo "export JAX_PLATFORM_NAME=cuda"
    else
        echo "export JAX_PLATFORM_NAME=cpu"
    fi
} >> /root/.bashrc

# Export locally for the current script session
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export JAX_PLATFORM_NAME=$([ "$MODE" = "gpu" ] && echo "cuda" || echo "cpu")

# --- 3. Python Stack Installation ---
echo "--- 3. Installing Libraries (NJU Mirror) ---"
MIRROR="https://mirror.nju.edu.cn/pypi/web/simple"
pip install --index-url $MIRROR --upgrade pip --break-system-packages

if [ "$MODE" = "gpu" ]; then
    echo "Installing GPU Version (jax[cuda12_pip])..."
    pip install --index-url $MIRROR --break-system-packages "jax[cuda12_pip]" mpi4py mpi4jax
else
    echo "Installing CPU Version (jax[cpu])..."
    pip install --index-url $MIRROR --break-system-packages "jax[cpu]" mpi4py mpi4jax
fi

# Core Physics & Visualization Libraries
pip install --index-url $MIRROR --break-system-packages \
    numpy scipy matplotlib jupyterlab ipywidgets jupyterlab_widgets \
    netket quspin physics-tenpy

echo ""
echo "===================================================="
echo "--- 4. PHYSICS VALIDATION ---"
echo "===================================================="

python3 -c "
import jax
import netket as nk
import quspin
backend = jax.default_backend()
print(f'✅ JAX Backend: {backend.upper()}')
print(f'✅ Devices found: {jax.devices()}')
print(f'✅ NetKet Version: {nk.__version__}')
print(f'✅ QuSpin Ready')
if '$MODE' == 'gpu' and backend != 'gpu':
    print('❌ ERROR: GPU requested but JAX is using CPU! Check LD_LIBRARY_PATH.')
"

# Test MPI distribution
echo "--- Testing MPI Distribution (Expect Rank 0 and 1) ---"
mpirun --allow-run-as-root -np 2 python3 -c "
import netket as nk
print(f'   [MPI] Rank {nk.utils.mpi.rank} verified')
"

echo "===================================================="
echo "✅ SETUP COMPLETE FOR ${MODE^^}!"
echo "===================================================="
echo "To access Jupyter Lab from your laptop:"
echo "1. Run on server: jupyter lab --allow-root --no-browser --port=8888"
echo "2. Run on laptop terminal: ssh -L 8888:localhost:8888 root@server-ip"
echo "3. Open browser: http://localhost:8888"
echo "===================================================="
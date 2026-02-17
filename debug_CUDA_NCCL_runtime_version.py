import ctypes
import glob
import os
import subprocess
from pathlib import Path

import torch


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def nccl_runtime_version():
    version = ctypes.c_int()
    try:
        lib = ctypes.CDLL("libnccl.so.2")
        rc = lib.ncclGetVersion(ctypes.byref(version))
        return rc, version.value
    except OSError as e:
        return None, f"OSError: {e}"


print("=== Torch/CUDA ===")
print(f"torch: {torch.__version__}")
print(f"torch cuda: {torch.version.cuda}")
print(f"torch nccl (reported): {torch.cuda.nccl.version() if torch.cuda.is_available() else None}")
print(f"torch path: {torch.__file__}")
print(f"CONDA_PREFIX: {os.getenv('CONDA_PREFIX')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print(f"LD_PRELOAD: {os.getenv('LD_PRELOAD')}")
print(f"NCCL_LIBRARY: {os.getenv('NCCL_LIBRARY')}")

print("\n=== NCCL Symlinks ===")
for p in sorted(glob.glob("/opt/conda/envs/gluefactory/lib/libnccl.so*")):
    rp = os.path.realpath(p)
    print(f"{p} -> {rp}")

print("\n=== NCCL Runtime (ctypes) ===")
rc, ver = nccl_runtime_version()
print(f"ncclGetVersion rc: {rc}")
print(f"ncclGetVersion value: {ver}")

print("\n=== Which libnccl is loaded by torch ===")
torch_c = Path(torch._C.__file__)
print(run(["bash", "-lc", f"ldd {torch_c} | grep -Ei 'nccl|cuda|cudart|c10'"]))

libtorch_cuda = Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"
if libtorch_cuda.exists():
    print(run(["bash", "-lc", f"ldd {libtorch_cuda} | grep -Ei 'nccl|cuda|cudart|c10'"]))

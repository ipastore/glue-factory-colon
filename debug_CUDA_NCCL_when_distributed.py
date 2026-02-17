import torch
import glob
import os
import subprocess


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    return out.strip()

print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Torch path: {torch.__file__}")
print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print(f"LD_PRELOAD: {os.getenv('LD_PRELOAD')}")
print(f"NCCL_LIBRARY: {os.getenv('NCCL_LIBRARY')}")
print(f"CONDA_PREFIX: {os.getenv('CONDA_PREFIX')}")

if torch.cuda.is_available():
    print(f"NCCL version: {torch.cuda.nccl.version()}")
else:
    print("CUDA is not available, cannot check NCCL version.")

p = os.path.join(os.path.dirname(torch.__file__), 'lib')

print("\n--- NCCL Libraries ---")
print('\n'.join(glob.glob(os.path.join(p, '*nccl*'))))

print("\n--- C10D Libraries ---")
print('\n'.join(glob.glob(os.path.join(p, '*c10d*'))))

print("\n--- ldd torch._C (cuda/nccl) ---")
print(_run(["bash", "-lc", f"ldd {torch._C.__file__} | grep -Ei 'cuda|nccl|cudart|c10'"]))

libtorch_cuda = os.path.join(p, "libtorch_cuda.so")
if os.path.exists(libtorch_cuda):
    print("\n--- ldd libtorch_cuda.so (cuda/nccl) ---")
    print(_run(["bash", "-lc", f"ldd {libtorch_cuda} | grep -Ei 'cuda|nccl|cudart|c10'"]))

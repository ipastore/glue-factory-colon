# Based on: https://github.com/cvg/glue-factory/issues/118
# Modified to save a generic_dict instead of an ordered_dict for loading weights in C++

import torch
from lightglue.lightglue import LightGlue
import gluefactory

if __name__ == "__main__":
    experiment = "longval001_null_10_lr1e-04"
    weights_name = "longval001_null_10_lr1e-04_600k.pt"
    
    # 1. Load the experiment
    print(f"Loading experiment: {experiment}")
    model = gluefactory.load_experiment(experiment)
    
    # 2. Extract state_dict
    state_dict = model.matcher.state_dict()
    
    # 3. CRITICAL FIX: Convert OrderedDict to standard dict
    # This strips the Python-specific "OrderedDict" class wrapper that confuses LibTorch
    plain_weights = dict(state_dict)
    
    # 4. CPU Enforcement: Ensure all tensors are on CPU
    # Saving CUDA tensors can cause issues if the C++ loader doesn't initialize CUDA exactly the same way
    for k, v in plain_weights.items():
        plain_weights[k] = v.cpu()

    # 5. Save with new zipfile serialization (Standard for LibTorch 1.6+)
    print(f"Saving to {weights_name}...")
    torch.save(plain_weights, weights_name, _use_new_zipfile_serialization=True)
    
    print("Success. This file is now safe for C++ 'pickle_load'.")
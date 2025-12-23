"""
A set of utilities to manage and load checkpoints of training experiments.

Author: Paul-Edouard Sarlin (skydes)
"""

import logging
import os
import re
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .. import settings
from ..models import get_model

logger = logging.getLogger(__name__)

def verify_checkpoint_loading(checkpoint, model, logger, module_prefix="matcher"):
    """Verify that checkpoint weights are loaded correctly into the model.
    
    Args:
        checkpoint: Dict containing the checkpoint with 'model' key
        model: The PyTorch model
        logger: Logger instance for output
        module_prefix: Prefix to check for specific module weights (default: "matcher")
    """
    state_dict = checkpoint["model"]
    loaded_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))

    # Check overlap
    matched = loaded_params & model_params
    logger.info(f"OK Successfully loaded {len(matched)}/{len(model_params)} parameters")

    # Check missing
    missing = model_params - loaded_params
    if missing:
        logger.warning(
            f" {len(missing)} parameters not in checkpoint (will be random):"
        )
        logger.warning(f"   Example: {list(missing)[:3]}")

    # Check extra (in checkpoint but not in model)
    extra = loaded_params - model_params
    if extra:
        logger.info(
            f"ℹ️  {len(extra)} extra parameters in checkpoint (ignored):"
        )
        logger.info(f"   Example: {list(extra)[:3]}")

    # Verify non-zero weights for specified module
    module_params = {
        k: v for k, v in state_dict.items() if k.startswith(f"{module_prefix}.")
    }
    if module_params:
        avg_abs = (
            sum(v.abs().mean().item() for v in module_params.values())
            / len(module_params)
        )
        logger.info(f"{module_prefix.capitalize()} weights average magnitude: {avg_abs:.6f}")
        if avg_abs < 1e-6:
            logger.error(f"{module_prefix.capitalize()} weights appear to be all zeros!")
    else:
        logger.error(f"No '{module_prefix}.' prefixed weights found in checkpoint!")
    
    return {
        "matched": len(matched),
        "total": len(model_params),
        "missing": len(missing),
        "extra": len(extra),
        "avg_magnitude": avg_abs if module_params else 0.0,
    }


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob("checkpoint_*.tar"):
        numbers = re.findall(r"(\d+)", p.name)
        assert len(numbers) <= 2
        if len(numbers) == 0:
            continue
        if len(numbers) == 1:
            checkpoints.append((int(numbers[0]), p))
        else:
            checkpoints.append((int(numbers[1]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    ckpts = list_checkpoints(Path(settings.TRAINING_PATH, exper))
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if "_interrupted" not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    p = Path(settings.TRAINING_PATH, exper, "checkpoint_best.tar")
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ("_interrupted" in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logger.info(f"Deleting checkpoint {ckpt[1].name}")
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(
    exper, conf={}, get_last=False, ckpt=None, weights_only=settings.ALLOW_PICKLE
):
    """Load and return the model of a given experiment."""
    exper = Path(exper)
    if exper.suffix != ".tar":
        if get_last:
            ckpt = get_last_checkpoint(exper)
        else:
            ckpt = get_best_checkpoint(exper)
    else:
        ckpt = exper
    logger.info(f"Loading checkpoint {ckpt.name}")
    ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=weights_only)

    loaded_conf = OmegaConf.create(ckpt["conf"])
    OmegaConf.set_struct(loaded_conf, False)
    conf = OmegaConf.merge(loaded_conf.model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()

    state_dict = ckpt["model"]
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))
    diff = model_params - dict_params
    if len(diff) > 0:
        subs = os.path.commonprefix(list(diff)).rstrip(".")
        logger.warning(f"Missing {len(diff)} parameters in {subs}")
    model.load_state_dict(state_dict, strict=False)
    return model


# @TODO: also copy the respective module scripts (i.e. the code)
def save_experiment(
    model,
    optimizer,
    lr_scheduler,
    conf,
    results,
    best_eval,
    epoch,
    iter_i,
    output_dir,
    stop=False,
    distributed=False,
    cp_name=None,
):
    """Save the current model to a checkpoint
    and return the best result so far."""
    state = (model.module if distributed else model).state_dict()
    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "conf": OmegaConf.to_container(conf, resolve=True),
        "epoch": epoch,
        "eval": results,
    }
    if cp_name is None:
        cp_name = (
            f"checkpoint_{epoch}_{iter_i}" + ("_interrupted" if stop else "") + ".tar"
        )
    logger.info(f"Saving checkpoint {cp_name}")
    cp_path = str(output_dir / cp_name)
    torch.save(checkpoint, cp_path)
    if cp_name != "checkpoint_best.tar" and results[conf.train.best_key] < best_eval:
        best_eval = results[conf.train.best_key]
        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
        shutil.copy(cp_path, str(output_dir / "checkpoint_best.tar"))
    delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
    return best_eval

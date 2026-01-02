import argparse
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..models import get_model
from ..utils.tensor import batch_to_device
from ..visualization.rotation_compare import make_rotation_compare_figures


def _build_model(conf, enable_rotation):
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    rot_conf = conf_dict.get("keypoint_rotation", {})
    rot_conf = deepcopy(rot_conf)
    rot_conf["enabled"] = enable_rotation
    rot_conf["train_only"] = False
    conf_dict["keypoint_rotation"] = rot_conf
    return get_model(conf_dict["name"])(conf_dict)


def _normalize_names(names, count):
    if names is None:
        return [f"pair_{i}" for i in range(count)]
    if torch.is_tensor(names):
        if names.ndim == 0:
            names = [names.item()] * count
        else:
            names = names.tolist()
    if isinstance(names, (list, tuple)):
        if len(names) < count:
            names = list(names) + [names[0]] * (count - len(names))
        names = [str(n) for n in names[:count]]
    else:
        names = [str(names)] * count
    return [n.replace("/", "_") for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="gluefactory/configs/CudaSift+lightglue_endomapper.yaml",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="outputs/rotation_debug")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(args.split, shuffle=False)

    batch = None
    for i, data in enumerate(loader):
        if i == args.batch_idx:
            batch = data
            break
    if batch is None:
        raise ValueError(f"Batch index {args.batch_idx} not found in loader.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = batch_to_device(batch, device, non_blocking=False)

    model_base = _build_model(conf.model, enable_rotation=False).to(device).eval()
    model_rot = _build_model(conf.model, enable_rotation=True).to(device).eval()
    model_rot.load_state_dict(model_base.state_dict(), strict=False)

    with torch.no_grad():
        pred_base = model_base(data)
        pred_rot = model_rot(data)

    rot_view = int(conf.model.get("keypoint_rotation", {}).get("view", 0))
    figs = make_rotation_compare_figures(
        pred_rot, pred_base, data, n_pairs=args.n_pairs, rot_view=rot_view
    )
    names = _normalize_names(data.get("names"), len(figs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fig, name in zip(figs, names):
        fig.savefig(
            out_dir / f"{name}_rotation_compare.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()

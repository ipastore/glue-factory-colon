#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from gluefactory.datasets.base_dataset import collate
from gluefactory.eval import get_benchmark
from gluefactory.eval.eval_pipeline import load_eval
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import EVAL_PATH
from gluefactory.visualization.two_view_frame import TwoViewFrame


def _decode_name(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _resolve_experiment_dir(benchmark: str, exp_dir: str) -> Path:
    p = Path(exp_dir)
    if p.exists():
        return p
    return Path(EVAL_PATH, benchmark, exp_dir)


def _load_index_from_results(experiment_dir: Path, pair_name: str):
    results_h5 = experiment_dir / "results.h5"
    if not results_h5.exists():
        return None
    _, results = load_eval(experiment_dir)
    if "names" not in results:
        return None
    names = [_decode_name(n) for n in results["names"]]
    return names.index(pair_name)


def _load_index_from_dataset(benchmark: str, pair_name: str):
    loader = get_benchmark(benchmark).get_dataloader()
    dataset = loader.dataset
    for i in range(len(dataset)):
        sample = dataset[i]
        name = _decode_name(sample["name"])
        if name == pair_name:
            return i
    raise ValueError(f"Pair not found in dataset: {pair_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect exactly one image pair from cached predictions."
    )
    parser.add_argument("benchmark", type=str)
    parser.add_argument(
        "exp_dir",
        type=str,
        help="Experiment name or full path to experiment dir containing predictions.h5",
    )
    parser.add_argument("pair_name", type=str, help="Pair key: image0/image1")
    parser.add_argument("--plot", type=str, default="matches")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = parser.parse_args()

    if args.backend:
        import matplotlib

        matplotlib.use(args.backend)

    experiment_dir = _resolve_experiment_dir(args.benchmark, args.exp_dir)
    pred_file = experiment_dir / "predictions.h5"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")

    try:
        idx = _load_index_from_results(experiment_dir, args.pair_name)
        if idx is None:
            idx = _load_index_from_dataset(args.benchmark, args.pair_name)
    except ValueError as e:
        raise ValueError(
            f"Could not find pair '{args.pair_name}'. Use the exact 'image0/image1' name from results.h5[names]."
        ) from e

    loader = get_benchmark(args.benchmark).get_dataloader()
    data = collate([loader.dataset[idx]])

    preds = {
        str(experiment_dir.name): CacheLoader(
            {"path": str(pred_file), "add_data_path": False}
        )(data)
    }

    conf = OmegaConf.create({"default": args.plot, "summary_visible": False})
    frame = TwoViewFrame(conf, data, preds, title=args.pair_name, event=1)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

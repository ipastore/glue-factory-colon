"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""

import argparse
import copy
import math
import re
import shutil
import signal
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
import matplotlib.image as mpimg
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import __module_name__, logger, settings
from .datasets import get_dataset
from .eval import run_benchmark
from .models import get_model
from .models.utils.metrics import matcher_metrics
from .visualization.gt_visualize_matches import (
    make_gt_pos_figs,
    make_gt_pos_neg_ign_figs,
)
from .utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment, verify_checkpoint_loading
from .utils.stdout_capturing import capture_outputs
from .utils.tensor import batch_to_device
from .utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)

# @TODO: Fix pbar pollution in logs
# @TODO: add plotting during evaluation

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
    "save_eval_figs": False,  # persist evaluation figures to disk
    "log_val_metrics": False,
    "log_overfit_metrics": False,
    "log_gt_pos_val_once": False,
    "log_gt_pos_overfit_once": False,
    "log_gt_pos_neg_ign_val_once": False,
    "log_gt_pos_neg_ign_overfit_once": False,
}
default_train_conf = OmegaConf.create(default_train_conf)


@torch.no_grad()
def do_evaluation(
    model,
    loader,
    device,
    loss_fn,
    conf,
    rank,
    pbar=True,
    baseline_model=None,
    plot_ids=None,
    baseline_preds=None,
    baseline_store=None,
    plot_kwargs=None,
    log_metrics_path=None,
    log_metrics_step=None,
):
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    results = {}
    pr_metrics = defaultdict(PRMetric)
    figures = []
    if plot_ids is None:
        plot_ids = []
    plot_fn = None
    plot_requires_oob = False
    if conf.plot is not None:
        n, plot_fn = conf.plot
        plot_ids = (
            plot_ids
            if len(plot_ids)
            else np.random.choice(len(loader), min(len(loader), n), replace=False)
        )
        plot_requires_oob = (
            "visualize_compare_lgoob.make_compare_lg_oob_figures" in plot_fn
        )
    log_file = None
    if log_metrics_path is not None and rank == 0:
        log_metrics_path = Path(log_metrics_path)
        write_header = not log_metrics_path.exists() or log_metrics_path.stat().st_size == 0
        log_file = log_metrics_path.open("a", encoding="ascii")
        if write_header:
            log_file.write(
                "step\tindex\tname\toverlap\tprecision\trecall\taccuracy\tap\n"
            )
    try:
        for i, data in enumerate(
            tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)
        ):
            data = batch_to_device(data, device, non_blocking=True)
            with torch.no_grad():
                pred = model(data)
            oob_pred = None
            if baseline_store is not None and i in list(plot_ids):
                baseline_store[i] = batch_to_device(pred, "cpu", non_blocking=False)
            losses, metrics = loss_fn(pred, data)
            if (
                conf.plot is not None
                and i in plot_ids
                and rank == 0
                and plot_fn is not None
            ):
                if plot_requires_oob and baseline_model is not None:
                    oob_pred = baseline_model(data)
                if plot_requires_oob and baseline_preds is not None:
                    oob_pred = baseline_preds.get(i, oob_pred)
                gt_values = {
                    k[len("gt_") :]: v for k, v in pred.items() if k.startswith("gt_")
                }
                plot_callable = locate(plot_fn)
                plot_result = (
                    plot_callable(
                        pred,
                        data,
                        pred_oob=oob_pred,
                        gt=gt_values,
                        **(plot_kwargs or {}),
                    )
                    if plot_requires_oob
                    else plot_callable(pred, data, **(plot_kwargs or {}))
                )
                if isinstance(plot_result, list):
                    figures.extend(plot_result)
                else:
                    figures.append(plot_result)
            if log_file is not None:
                gt_matches0 = pred.get("gt_matches0", data.get("gt_matches0"))
                overlap_vals = data.get("overlap_0to1")
                if gt_matches0 is not None and "matching_scores0" in pred:
                    per_item_metrics = matcher_metrics(
                        pred, {"gt_matches0": gt_matches0}
                    )
                    names = data.get("names")
                    if names is None:
                        names = [f"{i}_{j}" for j in range(gt_matches0.shape[0])]
                    elif torch.is_tensor(names):
                        names = (
                            names.tolist()
                            if names.ndim > 0
                            else [names.item()] * gt_matches0.shape[0]
                        )
                    elif not isinstance(names, (list, tuple)):
                        names = [names] * gt_matches0.shape[0]
                    for j in range(gt_matches0.shape[0]):
                        name = names[j] if j < len(names) else names[0]
                        overlap = (
                            overlap_vals[j].item()
                            if overlap_vals is not None
                            else float("nan")
                        )
                        line = (
                            f"{log_metrics_step}\t{i}_{j}\t{name}\t{overlap:.2f}\t"
                            f"{per_item_metrics['match_precision'][j].item():.6f}\t"
                            f"{per_item_metrics['match_recall'][j].item():.6f}\t"
                            f"{per_item_metrics['accuracy'][j].item():.6f}\t"
                            f"{per_item_metrics['average_precision'][j].item():.6f}\n"
                        )
                        log_file.write(line)
            # add PR curves
            for k, v in conf.pr_curves.items():
                pr_metrics[k].update(
                    pred[v["labels"]],
                    pred[v["predictions"]],
                    mask=pred[v["mask"]] if "mask" in v.keys() else None,
                )
            del pred, data
            numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
            for k, v in numbers.items():
                if k not in results:
                    results[k] = AverageMetric()
                    if k in conf.median_metrics:
                        results[k + "_median"] = MedianMetric()
                    if k in conf.recall_metrics.keys():
                        q = conf.recall_metrics[k]
                        results[k + f"_recall{int(q)}"] = RecallMetric(q)
                results[k].update(v)
                if k in conf.median_metrics:
                    results[k + "_median"].update(v)
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"].update(v)
            del numbers
    finally:
        if log_file is not None:
            log_file.close()
    results = {k: results[k].compute() for k in results}
    pr_metrics = {k: v.compute() for k, v in pr_metrics.items()}
    return results, pr_metrics, figures


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", None]:
        if hasattr(conf.options, "schedulers"):
            # Add option to chain multiple schedulers together
            # This is useful for e.g. warmup, then cosine decay
            schedulers = []
            for scheduler_conf in conf.options.schedulers:
                scheduler = get_lr_scheduler(optimizer, scheduler_conf)
                schedulers.append(scheduler)

            options = {k: v for k, v in conf.options.items() if k != "schedulers"}
            return getattr(torch.optim.lr_scheduler, conf.type)(
                optimizer, schedulers, **options
            )

        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]}
        for scale, ps in scale2params.items()
    ]
    return lr_params


def write_dict_summaries(writer, name, items, step):
    for k, v in items.items():
        key = f"{name}/{k}"
        if isinstance(v, dict):
            writer.add_scalars(key, v, step)
        elif isinstance(v, tuple):
            writer.add_pr_curve(key, *v, step)
        else:
            writer.add_scalar(key, v, step)


def write_image_summaries(writer, name, figures, step):
    if isinstance(figures, list):
        for i, figs in enumerate(figures):
            for k, fig in figs.items():
                writer.add_figure(f"{name}/{i}_{k}", fig, step)
    else:
        for k, fig in figures.items():
            writer.add_figure(f"{name}/{k}", fig, step)


def generate_gt_figures(model, batch, device, fig_fn, pos_th=None, neg_th=None):
    batch = batch_to_device(batch, device, non_blocking=True)
    if "image" not in batch["view0"] or "image" not in batch["view1"]:
        return []
    with torch.no_grad():
        pred = model(batch)
        model.loss(pred, batch)
    gt_values = {k[len("gt_") :]: v for k, v in pred.items() if k.startswith("gt_")}
    if not gt_values:
        return []
    if "keypoints0" not in batch:
        view0_cache = batch["view0"].get("cache", {})
        view1_cache = batch["view1"].get("cache", {})
        batch = {
            **batch,
            "keypoints0": view0_cache.get("keypoints"),
            "keypoints1": view1_cache.get("keypoints"),
            "valid_3D_mask0": view0_cache.get("valid_3D_mask"),
            "valid_3D_mask1": view1_cache.get("valid_3D_mask"),
            "keypoint_scores0": view0_cache.get("keypoint_scores"),
            "keypoint_scores1": view1_cache.get("keypoint_scores"),
        }
    fig_kwargs = {"n_pairs": batch["keypoints0"].shape[0]}
    if fig_fn is make_gt_pos_neg_ign_figs:
        fig_kwargs["pos_th"] = pos_th
        fig_kwargs["neg_th"] = neg_th
        if "matches0" not in batch and "matches0" in gt_values:
            batch = {**batch, "matches0": gt_values["matches0"].long()}
        if "matches1" not in batch and "matches1" in gt_values:
            batch = {**batch, "matches1": gt_values["matches1"].long()}
    return fig_fn(gt_values, batch, **fig_kwargs)


def _clean_plot_name(name):
    s = str(name)
    s = s.replace("Seq_", "")
    s = s.replace("Keyframe_", "")
    for ext in [".png", ".pg"]:
        s = s.replace(ext, "")
    s = s.replace("/", "_")
    return s


def log_gt_pos_figures(writer, figures, save_dir, tag_prefix, step, names=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx, fig in enumerate(figures):
        if names:
            raw_name = names[idx] if idx < len(names) else names[0]
            name = f"{tag_prefix}_{_clean_plot_name(raw_name)}"
        else:
            name = f"{tag_prefix}_{idx}"
        fig_path = save_dir / f"{name}.png"
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=0, dpi=300)
        img = mpimg.imread(fig_path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        tag = f"{tag_prefix}/{_clean_plot_name(raw_name)}" if names else f"{tag_prefix}/{idx}"
        writer.add_image(tag, img, step, dataformats="HWC")


def collect_batches_by_ids(loader, ids):
    ids = set(ids)
    batches = []
    for idx, batch in enumerate(loader):
        if idx in ids:
            batches.append((idx, batch))
            if len(batches) == len(ids):
                break
    return batches


def save_eval_figures(figures, save_dir: Path, prefix: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    dpi = 300
    if isinstance(figures, list):
        for i, figs in enumerate(figures):
            if isinstance(figs, dict):
                for k, fig in figs.items():
                    fig.savefig(
                        save_dir / f"{prefix}_{i}_comparison.png",
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=dpi,
                    )
            else:
                figs.savefig(
                    save_dir / f"{prefix}_{i}_comparison.png",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=dpi,
                )
    elif isinstance(figures, dict):
        for k, fig in figures.items():
            fig.savefig(
                save_dir / f"{prefix}_comparison.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=dpi,
            )
    else:
        figures.savefig(
            save_dir / f"{prefix}_comparison.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )


def training(rank, conf, output_dir, args):
    if args.restore:
        logger.info(f"Restoring from previous training of {args.experiment}")
        try:
            init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        except AssertionError:
            init_cp = get_best_checkpoint(args.experiment)
        logger.info(f"Restoring from checkpoint {init_cp.name}")
        init_cp = torch.load(
            str(init_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
        )
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = init_cp["epoch"] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(
            str(best_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
        )
        best_eval = best_cp["eval"][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float("inf")
        if conf.train.load_experiment:
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            # the user has to make sure that the weights are compatible
            try:
                init_cp = get_last_checkpoint(conf.train.load_experiment)
            except AssertionError:
                init_cp = get_best_checkpoint(conf.train.load_experiment)
            # init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(
                str(init_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
            )
            # load the model config of the old setup, and overwrite with current config
            conf.model = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).model, conf.model
            )
            print(conf.model)

        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f"Training in distributed mode with {args.n_gpus} GPUs")
        assert torch.cuda.is_available()
        device = rank
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.n_gpus,
            rank=device,
            init_method="file://" + str(args.lock_file),
        )
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if "batch_size" in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if "train_batch_size" in data_conf:
            data_conf.train_batch_size = int(data_conf.train_batch_size / args.n_gpus)
        if "num_workers" in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus
            )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}")

    dataset = get_dataset(data_conf.name)(data_conf)

    # Optionally load a different validation dataset than the training one
    val_data_conf = conf.get("data_val", None)
    if val_data_conf is None:
        val_dataset = dataset
    else:
        val_dataset = get_dataset(val_data_conf.name)(val_data_conf)

    # @TODO: add test data loader

    if args.overfit:
        # we train and eval with the same single training batch
        logger.info("Data in overfitting mode")
        assert not args.distributed
        train_loader = dataset.get_overfit_loader("train")
        val_loader = val_dataset.get_overfit_loader("val")
    else:
        train_loader = dataset.get_data_loader("train", distributed=args.distributed)
        val_loader = val_dataset.get_data_loader("val")
    overfit_loader = None
    if not args.overfit:
        overfit_split = val_dataset.conf.get("overfit_split", None)
        if overfit_split is not None:
            overfit_loader = val_dataset.get_data_loader("overfit")
    if rank == 0:
        logger.info(f"Training loader has {len(train_loader)} batches")
        logger.info(f"Validation loader has {len(val_loader)} batches")
        if overfit_loader is not None:
            logger.info(f"Overfit loader has {len(overfit_loader)} batches")

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logger.info("Caught keyboard interrupt signal, will terminate")
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True

    stop = False
    signal.signal(signal.SIGINT, sigint_handler)
    model = get_model(conf.model.name)(conf.model).to(device)
    if init_cp is not None:
        model.load_state_dict(init_cp["model"], strict=False)
        # Verify checkpoint loading
        verify_checkpoint_loading(init_cp, model, logger, module_prefix="matcher")
    loss_fn = model.loss
    requires_oob_plot = bool(
        conf.train.plot
        and "visualize_compare_lgoob.make_compare_lg_oob_figures"
        in conf.train.plot[1]
    )
    plot_ids_static = None
    plot_ids_overfit = None
    baseline_preds_cache = None
    baseline_preds_overfit_cache = None
    baseline_ready = False
    gt_pos_val_logged = False
    gt_pos_overfit_logged = False
    gt_pos_neg_ign_val_logged = False
    gt_pos_neg_ign_overfit_logged = False
    if requires_oob_plot:
        pos_th = conf.model.ground_truth.get("th_positive", None)
        neg_th = conf.model.ground_truth.get("th_negative", None)
        n, _ = conf.train.plot
        plot_ids_static = np.random.choice(
            len(val_loader), min(len(val_loader), n), replace=False
        )
        if rank == 0:
            logger.info("Running initial val eval to cache OOB predictions for plotting.")
            baseline_preds_cache = {}
            baseline_conf = copy.deepcopy(conf.train)
            baseline_conf.plot = None
            baseline_conf.save_eval_figs = False
            with fork_rng(seed=conf.train.seed):
                do_evaluation(
                    model,
                    val_loader,
                    device,
                    loss_fn,
                    baseline_conf,
                    rank,
                    pbar=(rank == 0),
                    plot_ids=plot_ids_static,
                    baseline_store=baseline_preds_cache,
                )
            val_batches = collect_batches_by_ids(val_loader, plot_ids_static)
            if conf.train.log_gt_pos_val_once and not gt_pos_val_logged:
                for batch_idx, val_batch in val_batches:
                    names = val_batch.get("names")
                    if torch.is_tensor(names):
                        names = (
                            names.tolist() if names.ndim > 0 else [names.item()]
                        )
                    elif names is not None and not isinstance(names, (list, tuple)):
                        names = [names]
                    gt_pos_figs = generate_gt_figures(
                        model, val_batch, device, make_gt_pos_figs
                    )
                    log_gt_pos_figures(
                        writer,
                        gt_pos_figs,
                        output_dir / "gt_pos_val",
                        "gt_pos_val",
                        0,
                        names=names,
                    )
                gt_pos_val_logged = True
            if (
                conf.train.log_gt_pos_neg_ign_val_once
                and not gt_pos_neg_ign_val_logged
            ):
                for batch_idx, val_batch in val_batches:
                    names = val_batch.get("names")
                    if torch.is_tensor(names):
                        names = (
                            names.tolist() if names.ndim > 0 else [names.item()]
                        )
                    elif names is not None and not isinstance(names, (list, tuple)):
                        names = [names]
                    gt_pos_neg_ign_figs = generate_gt_figures(
                        model,
                        val_batch,
                        device,
                        make_gt_pos_neg_ign_figs,
                        pos_th=pos_th,
                        neg_th=neg_th,
                    )
                    log_gt_pos_figures(
                        writer,
                        gt_pos_neg_ign_figs,
                        output_dir / "gt_pos_neg_ign_val",
                        "gt_pos_neg_ign_val",
                        0,
                        names=names,
                    )
                gt_pos_neg_ign_val_logged = True
            if overfit_loader is not None:
                logger.info("Running initial overfit eval to cache OOB predictions for plotting.")
                plot_ids_overfit = np.random.choice(
                    len(overfit_loader), min(len(overfit_loader), n), replace=False
                )
                baseline_preds_overfit_cache = {}
                with fork_rng(seed=conf.train.seed):
                    do_evaluation(
                        model,
                        overfit_loader,
                        device,
                        loss_fn,
                        baseline_conf,
                        rank,
                        pbar=(rank == 0),
                        plot_ids=plot_ids_overfit,
                        baseline_store=baseline_preds_overfit_cache,
                    )
                overfit_batches = collect_batches_by_ids(
                    overfit_loader, plot_ids_overfit
                )
                if conf.train.log_gt_pos_overfit_once and not gt_pos_overfit_logged:
                    for batch_idx, overfit_batch in overfit_batches:
                        names = overfit_batch.get("names")
                        if torch.is_tensor(names):
                            names = (
                                names.tolist() if names.ndim > 0 else [names.item()]
                            )
                        elif names is not None and not isinstance(names, (list, tuple)):
                            names = [names]
                        gt_pos_figs = generate_gt_figures(
                            model, overfit_batch, device, make_gt_pos_figs
                        )
                        log_gt_pos_figures(
                            writer,
                            gt_pos_figs,
                            output_dir / "gt_pos_overfit",
                            "gt_pos_overfit",
                            0,
                            names=names,
                        )
                    gt_pos_overfit_logged = True
                if (
                    conf.train.log_gt_pos_neg_ign_overfit_once
                    and not gt_pos_neg_ign_overfit_logged
                ):
                    for batch_idx, overfit_batch in overfit_batches:
                        names = overfit_batch.get("names")
                        if torch.is_tensor(names):
                            names = (
                                names.tolist() if names.ndim > 0 else [names.item()]
                            )
                        elif names is not None and not isinstance(names, (list, tuple)):
                            names = [names]
                        gt_pos_neg_ign_figs = generate_gt_figures(
                            model,
                            overfit_batch,
                            device,
                            make_gt_pos_neg_ign_figs,
                            pos_th=pos_th,
                            neg_th=neg_th,
                        )
                        log_gt_pos_figures(
                            writer,
                            gt_pos_neg_ign_figs,
                            output_dir / "gt_pos_neg_ign_overfit",
                            "gt_pos_neg_ign_overfit",
                            0,
                            names=names,
                        )
                    gt_pos_neg_ign_overfit_logged = True
        baseline_ready = True
    if args.compile:
        model = torch.compile(model, mode=args.compile)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    if rank == 0 and args.print_arch:
        logger.info(f"Model: \n{model}")

    torch.backends.cudnn.benchmark = True
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
        lr_params, lr=conf.train.lr, **conf.train.optimizer_options
    )
    use_mp = args.mixed_precision is not None
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_mp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_mp)
    )
    logger.info(f"Training with mixed_precision={args.mixed_precision}")

    mp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: torch.float32,  # we disable it anyway
    }[args.mixed_precision]

    results = None  # fix bug with it saving

    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)
    if args.restore:
        optimizer.load_state_dict(init_cp["optimizer"])
        if "lr_scheduler" in init_cp:
            lr_scheduler.load_state_dict(init_cp["lr_scheduler"])

    if rank == 0:
        logger.info(
            "Starting training with configuration:\n%s", OmegaConf.to_yaml(conf)
        )

    def trace_handler(p):
        # torch.profiler.tensorboard_trace_handler(str(output_dir))
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
        p.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        # we first run the eval
        if (
            rank == 0
            and epoch % conf.train.test_every_epoch == 0
            and args.run_benchmarks
        ):
            for bname, eval_conf in conf.get("benchmarks", {}).items():
                logger.info(f"Running eval on {bname}")
                summaries, figures, _ = run_benchmark(
                    bname,
                    eval_conf,
                    settings.EVAL_PATH / bname / args.experiment / str(epoch),
                    model.eval(),
                )
                str_summaries = [
                    f"{k} {v:.3E}" for k, v in summaries.items() if isinstance(v, float)
                ]
                logger.info(f'[{bname}] {{{", ".join(str_summaries)}}}')
                write_dict_summaries(writer, f"test/{bname}", summaries, epoch)
                write_image_summaries(writer, f"figures/{bname}", figures, epoch)
                del summaries, figures

        # set the seed
        set_seed(conf.train.seed + epoch)

        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logger.info(
                f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}'
            )
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn and not args.overfit:
            loaders = [train_loader]
            if conf.train.dataset_callback_on_val:
                loaders += [val_loader]
            for loader in loaders:
                if isinstance(loader.dataset, torch.utils.data.Subset):
                    getattr(loader.dataset.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
                else:
                    getattr(loader.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
        if rank == 0:
            def _log_loader_stats(name, loader):
                size = len(loader.dataset)
                bs = getattr(loader, "batch_size", None)
                drop_last = getattr(loader, "drop_last", False)
                iter_per_epoch = size / bs
                logger.info(
                    "%s epoch stats: size=%d, batch_size=%s, drop_last=%s, "
                    "loader_batches=%d, iter_per_epoch=%.2f",
                    name,
                    size,
                    bs,
                    drop_last,
                    len(loader),
                    iter_per_epoch,
                )
            _log_loader_stats("Train", train_loader)
            _log_loader_stats("Val", val_loader)
            if overfit_loader is not None:
                _log_loader_stats("Overfit", overfit_loader)
        for it, data in enumerate(train_loader):
            tot_it = (len(train_loader) * epoch + it) * (
                args.n_gpus if args.distributed else 1
            )
            tot_n_samples = tot_it
            if not args.log_it:
                # We normalize the x-axis of tensorflow to num samples!
                tot_n_samples *= train_loader.batch_size

            model.train()
            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=args.mixed_precision is not None,
                dtype=mp_dtype,
            ):
                data = batch_to_device(data, device, non_blocking=True)
                pred = model(data)
                losses, _ = loss_fn(pred, data)
                loss = torch.mean(losses["total"])
            if torch.isnan(loss).any():
                print(f"Detected NAN, skipping iteration {it}")
                del pred, data, loss, losses
                continue

            do_backward = loss.requires_grad
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(
                    do_backward, torch.distributed.ReduceOp.PRODUCT
                )
                do_backward = do_backward > 0
            if do_backward:
                scaler.scale(loss).backward()
                if args.detect_anomaly:
                    # Check for params without any gradient which causes
                    # problems in distributed training with checkpointing
                    detected_anomaly = False
                    for name, param in model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            print(f"param {name} has no gradient.")
                            detected_anomaly = True
                    if detected_anomaly:
                        raise RuntimeError("Detected anomaly in training.")
                if conf.train.get("clip_grad", None):
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            all_params,
                            max_norm=conf.train.clip_grad,
                            error_if_nonfinite=True,
                        )
                        scaler.step(optimizer)
                    except RuntimeError:
                        logger.warning("NaN detected in gradients. Skipping iteration.")
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                if not conf.train.lr_schedule.on_epoch:
                    lr_scheduler.step()
            else:
                if rank == 0:
                    logger.warning(f"Skip iteration {it} due to detach.")

            if args.profile:
                prof.step()

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum(-1)
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= train_loader.batch_size * args.n_gpus
                    losses[k] = torch.mean(losses[k], -1)
                    losses[k] = losses[k].item()
                if rank == 0:
                    str_losses = [f"{k} {v:.3E}" for k, v in losses.items()]
                    logger.info(
                        "[E {} | it {}] loss {{{}}}".format(
                            epoch, it, ", ".join(str_losses)
                        )
                    )
                    write_dict_summaries(writer, "training/", losses, tot_n_samples)
                    writer.add_scalar(
                        "training/lr", optimizer.param_groups[0]["lr"], tot_n_samples
                    )
                    writer.add_scalar("training/epoch", epoch, tot_n_samples)

            if conf.train.log_grad_every_iter is not None:
                if it % conf.train.log_grad_every_iter == 0:
                    grad_txt = ""
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            if name.endswith("bias"):
                                continue
                            writer.add_histogram(
                                f"grad/{name}", param.grad.detach(), tot_n_samples
                            )
                            norm = torch.norm(param.grad.detach(), 2)
                            grad_txt += f"{name} {norm.item():.3f}  \n"
                    writer.add_text("grad/summary", grad_txt, tot_n_samples)
            del pred, data, loss, losses

            # Run validation
            run_eval = (
                (
                    it % conf.train.eval_every_iter == 0
                    and (it > 0 or epoch == -int(args.no_eval_0))
                )
                or stop
                or it == (len(train_loader) - 1)
            )
            if run_eval and not (baseline_ready and epoch == 0 and it == 0):
                with fork_rng(seed=conf.train.seed):
                    results, pr_metrics, figures = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        rank,
                        pbar=(rank == 0),
                        baseline_preds=baseline_preds_cache,
                        plot_ids=plot_ids_static,
                        plot_kwargs=(
                            {"epoch_idx": epoch} if conf.train.plot is not None else None
                        ),
                        log_metrics_path=(
                            output_dir / "val_metrics.txt"
                            if conf.train.log_val_metrics
                            else None
                        ),
                        log_metrics_step=tot_n_samples,
                    )

                if rank == 0:
                    str_results = [
                        f"{k} {v:.3E}"
                        for k, v in results.items()
                        if isinstance(v, float)
                    ]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')
                    write_dict_summaries(writer, "val", results, tot_n_samples)
                    write_dict_summaries(writer, "val", pr_metrics, tot_n_samples)
                    write_image_summaries(writer, "figures_val", figures, tot_n_samples)
                    if conf.train.save_eval_figs:
                        save_dir = output_dir / "eval_figs"
                        save_eval_figures(
                            figures, save_dir, f"E{epoch}"
                        )
                    # @TODO: optional always save checkpoint
                    if results[conf.train.best_key] < best_eval:
                        best_eval = results[conf.train.best_key]
                        save_experiment(
                            model,
                            optimizer,
                            lr_scheduler,
                            conf,
                            results,
                            best_eval,
                            epoch,
                            tot_it,
                            output_dir,
                            stop,
                            args.distributed,
                            cp_name="checkpoint_best.tar",
                        )
                        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
                if overfit_loader is not None:
                    with fork_rng(seed=conf.train.seed):
                        overfit_results, overfit_pr_metrics, overfit_figures = do_evaluation(
                            model,
                            overfit_loader,
                            device,
                            loss_fn,
                            conf.train,
                            rank,
                            pbar=(rank == 0),
                            baseline_preds=baseline_preds_overfit_cache,
                            plot_ids=plot_ids_overfit,
                            plot_kwargs=(
                                {"epoch_idx": epoch}
                                if conf.train.plot is not None
                                else None
                            ),
                            log_metrics_path=(
                                output_dir / "overfit_metrics.txt"
                                if conf.train.log_overfit_metrics
                                else None
                            ),
                            log_metrics_step=tot_n_samples,
                        )
                    if rank == 0:
                        str_overfit_results = [
                            f"{k} {v:.3E}"
                            for k, v in overfit_results.items()
                            if isinstance(v, float)
                        ]
                        logger.info(f'[Overfit] {{{", ".join(str_overfit_results)}}}')
                        write_dict_summaries(
                            writer, "overfit", overfit_results, tot_n_samples
                        )
                        write_dict_summaries(
                            writer, "overfit", overfit_pr_metrics, tot_n_samples
                        )
                        write_image_summaries(
                            writer,
                            "figures_overfit",
                            overfit_figures,
                            tot_n_samples,
                        )
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if (tot_it % conf.train.save_every_iter == 0 and tot_it > 0) and rank == 0:
                if results is None:
                    results, _, _ = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        rank,
                        pbar=(rank == 0),
                        baseline_preds=baseline_preds_cache,
                        plot_ids=plot_ids_static,
                        plot_kwargs=(
                            {"epoch_idx": epoch} if conf.train.plot is not None else None
                        ),
                    )
                    best_eval = results[conf.train.best_key]
                best_eval = save_experiment(
                    model,
                    optimizer,
                    lr_scheduler,
                    conf,
                    results,
                    best_eval,
                    epoch,
                    tot_it,
                    output_dir,
                    stop,
                    args.distributed,
                )
            if stop:
                break

        if rank == 0:
            best_eval = save_experiment(
                model,
                optimizer,
                lr_scheduler,
                conf,
                results,
                best_eval,
                epoch,
                tot_it,
                output_dir=output_dir,
                stop=stop,
                distributed=args.distributed,
            )

        results = None  # free memory
        epoch += 1

    logger.info(f"Finished training on process {rank}.")
    if rank == 0:
        writer.close()


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(
            output_dir / "log.txt", cleanup_interval=args.cleanup_interval
        ):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--cleanup_interval",
        default=120,  # Cleanup log files every 120 seconds.
        type=int,
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment {args.experiment}")
    output_dir = Path(settings.TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        yaml_conf = OmegaConf.load(args.conf)
        OmegaConf.resolve(yaml_conf)
        conf = OmegaConf.merge(yaml_conf, conf)
    elif args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
    conf = OmegaConf.merge(conf, {"train": {"experiment_name": args.experiment}})

    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))

    # copy gluefactory and submodule into output dir
    for module in conf.train.get("submodules", []) + [__module_name__]:
        mod_dir = Path(__import__(str(module)).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)
    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(conf, output_dir, args)
        )
    else:
        main_worker(0, conf, output_dir, args)

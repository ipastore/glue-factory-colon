#!/usr/bin/env python3
"""Aggregate MegaDepth eval summaries into a single table."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/results/megadepth1500")
DEFAULT_COLUMNS = [
    "experiment",
    "num_total_pairs",
    "mepi_prec@1e-4",
    "mepi_prec@5e-4",
    "mepi_prec@1e-3",
    "valid_pairs_epi",
    "mnum_matches",
    "mnum_keypoints",
    "mreproj_prec@1px",
    "mreproj_prec@3px",
    "mreproj_prec@5px",
    "valid_pairs_reproj",
    "mcovisible",
    "mcovisible_percent",
    "mgt_match_recall@3px",
    "mgt_match_precision@3px",
    "rel_pose_error@5°",
    "rel_pose_error@10°",
    "rel_pose_error@20°",
    "rel_pose_error_mAA",
    "mrel_pose_error",
    "valid_pairs_pose",
    "mransac_inl",
    "mransac_inl%",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single report table from MegaDepth summaries.json files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory that contains one folder per experiment (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "csv", "md"),
        default="plain",
        help="Output table format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional file path to write the table. "
            "If omitted, writes under --root with an auto-generated filename."
        ),
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help=(
            "Deprecated (kept for compatibility): all metrics are now included by default."
        ),
    )
    parser.add_argument(
        "--default-metrics",
        action="store_true",
        help="Use a curated metric subset instead of all metrics.",
    )
    parser.add_argument(
        "--split-name",
        action="store_true",
        help=(
            "Add parsed columns from experiment folder name: extractor, extractor_conf, matcher_conf."
        ),
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include experiments without summaries.json (metrics left blank).",
    )
    parser.add_argument(
        "--sort-by",
        default="experiment",
        help="Column used to sort rows (default: experiment).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_experiment_name(name: str) -> dict[str, str]:
    extractor = ""
    extractor_conf = ""
    matcher_conf = ""

    # Pattern 1: "<extractor>_params_<extractor_conf>-<matcher_conf>"
    if "_params_" in name and "-" in name:
        left, matcher_conf = name.split("-", 1)
        extractor, extractor_conf = left.split("_params_", 1)
        return {
            "extractor": extractor,
            "extractor_conf": extractor_conf,
            "matcher_conf": matcher_conf,
        }

    # Pattern 2: "<extractor>+<matcher_conf>"
    if "+" in name:
        extractor, matcher_conf = name.split("+", 1)
        return {
            "extractor": extractor,
            "extractor_conf": "",
            "matcher_conf": matcher_conf,
        }

    return {
        "extractor": name,
        "extractor_conf": "",
        "matcher_conf": "",
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def build_rows(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary_path = exp_dir / "summaries.json"
        if not summary_path.exists() and not args.include_missing:
            continue

        row: dict[str, Any] = {"experiment": exp_dir.name}
        if args.split_name:
            row.update(parse_experiment_name(exp_dir.name))

        if summary_path.exists():
            summary = load_summary(summary_path)
            row.update(summary)
            metric_keys.update(summary.keys())

        rows.append(row)

    use_all_metrics = not args.default_metrics

    if use_all_metrics:
        # Stable order: preferred keys first, then remaining keys alphabetically.
        preferred = [k for k in DEFAULT_COLUMNS if k != "experiment"]
        rest = sorted(k for k in metric_keys if k not in set(preferred))
        columns = []
        if args.split_name:
            columns.extend(["extractor", "extractor_conf", "matcher_conf"])
        else:
            columns.append("experiment")
        columns.extend([k for k in preferred if k in metric_keys])
        columns.extend(rest)
    else:
        columns = []
        if args.split_name:
            columns.extend(["extractor", "extractor_conf", "matcher_conf"])
        else:
            columns.append("experiment")
        columns.extend([k for k in DEFAULT_COLUMNS if k != "experiment"])

    return columns, rows


def sort_rows(rows: list[dict[str, Any]], column: str, descending: bool) -> None:
    def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
        value = row.get(column, None)
        if value is None:
            return (1, "")
        return (0, value)

    rows.sort(key=sort_key, reverse=descending)


def render_plain(columns: list[str], rows: list[dict[str, Any]]) -> str:
    data = [[format_value(row.get(c, "")) for c in columns] for row in rows]
    widths = [len(c) for c in columns]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "-+-".join("-" * widths[i] for i in range(len(columns)))
    lines = [header, sep]
    for row in data:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))
    return "\n".join(lines)


def render_md(columns: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = [format_value(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    out = []
    out.append(",".join(columns))
    for row in rows:
        out.append(",".join(format_value(row.get(c, "")) for c in columns))
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    columns, rows = build_rows(args)
    sort_rows(rows, args.sort_by, args.descending)

    if args.format == "plain":
        output = render_plain(columns, rows)
    elif args.format == "md":
        output = render_md(columns, rows)
    else:
        output = render_csv(columns, rows)

    if args.output is None:
        suffix = {"csv": "csv", "md": "md", "plain": "txt"}[args.format]
        args.output = args.root / f"summaries_table.{suffix}"

    if args.format == "csv":
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([format_value(row.get(c, "")) for c in columns])
    else:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()

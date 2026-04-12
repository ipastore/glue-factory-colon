#!/usr/bin/env python3
"""Extract Endomapper ROMA pair names from a metrics txt file."""

import argparse
import re
from pathlib import Path


PAIR_RE = re.compile(
    r"(Seq_[^/\s]+/Keyframe_\d+\.png_Keyframe_\d+\.png)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Seq_.../Keyframe_....png_Keyframe_....png pairs from a metrics txt file."
    )
    parser.add_argument("input_txt", type=Path, help="Path to the input metrics txt file.")
    parser.add_argument("output_txt", type=Path, help="Path to the output pairs txt file.")
    return parser.parse_args()


def main():
    args = parse_args()
    lines = args.input_txt.read_text(encoding="ascii").splitlines()

    pairs = []
    for line in lines:
        match = PAIR_RE.search(line)
        if match is not None:
            pairs.append(match.group(1))

    output = "\n".join(pairs)
    if output:
        output += "\n"

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(output, encoding="ascii")
    print(f"Extracted {len(pairs)} pairs to {args.output_txt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <mov_root> <out_root>" >&2
  exit 1
fi

mov_root=$1
out_root=$2
W=720
H=540

shopt -s nullglob
for seq_dir in "$mov_root"/Seq_*; do
  [ -d "$seq_dir" ] || continue
  seq_name=$(basename "$seq_dir")
  video_path="$seq_dir/$seq_name.mov"
  if [ ! -f "$video_path" ]; then
    echo "[skip] missing video: $video_path" >&2
    continue
  fi
  echo "Processing $video_path"
  outdir="$out_root/$seq_name"
  mkdir -p "$outdir"
  ffmpeg -hide_banner -loglevel error -y -i "$video_path" -vf "scale=${W}:${H}" -vsync 0 -start_number 0 \
    "$outdir/Frame%08d.png"
done

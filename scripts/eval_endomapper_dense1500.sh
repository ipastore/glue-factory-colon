#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_MODULE="gluefactory.eval.endomapper_dense1500"

run_eval() {
    local tag="$1"
    local conf="$2"
    shift 2
    echo "==> ${tag}"
    "${PYTHON_BIN}" -m "${EVAL_MODULE}" --conf "${conf}" --tag "${tag}" "$@" "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

# run_eval "superpoint_official+nn" "superpoint+NN"
# run_eval "superpoint_official+lightglue_official" "superpoint+lightglue-official"
# run_eval \
#     "superpoint_official+roma" "roma" \
#     model.extractor.name=gluefactory_nonfree.superpoint \
#     model.extractor.max_num_keypoints=2048 \
#     model.extractor.detection_threshold=0.0 \
#     model.extractor.nms_radius=0 \
#     model.matcher.sample_num_matches=0

# run_eval \
#     "sift_opencv+nn" "sift+NN" \
#     model.extractor.backend=opencv \
#     model.extractor.max_num_keypoints=4096
# run_eval \
#     "sift_pycolmap+nn" "sift+NN" \
#     model.extractor.backend=pycolmap_cuda \
#     model.extractor.max_num_keypoints=4096
# run_eval \
#     "sift_py_cudasift+nn" "sift+NN" \
#     model.extractor.backend=py_cudasift \
#     model.extractor.max_num_keypoints=4096

# run_eval \
#     "sift_opencv+lightglue_official" "eval_pycolmap+lightglue-official" \
#     model.extractor.backend=opencv \
#     model.extractor.max_num_keypoints=4096
# # run_eval "sift_pycolmap+lightglue_official" "eval_pycolmap+lightglue-official"
# run_eval "sift_py_cudasift+lightglue_official" "eval_py_cudasift+lightglue-official"

# # run_eval \
# #     "sift_opencv+lightglue_finetuned" "eval_pycolmap+lightglue-finetuned" \
# #     model.extractor.backend=opencv
# # run_eval "sift_pycolmap+lightglue_finetuned" "eval_pycolmap+lightglue-finetuned"
# run_eval \
#     "sift_opencv+lightglue_finetuned" "eval_py_cudasift+lightglue-finetuned" \
#     model.extractor.backend=opencv
# run_eval "sift_py_cudasift+lightglue_finetuned" "eval_py_cudasift+lightglue-finetuned"

# run_eval \
#     "sift_opencv+roma" "roma" \
#     model.extractor.name=extractors.sift \
#     model.extractor.backend=opencv \
#     model.extractor.max_num_keypoints=4096 \
#     model.extractor.detection_threshold=0.0066667 \
#     model.extractor.rootsift=true \
#     model.extractor.nms_radius=0 \
#     model.extractor.first_octave=-1 \
#     model.extractor.num_octaves=4 \
#     model.extractor.init_blur=1.0 \
#     model.extractor.force_num_keypoints=False \
#     model.matcher.sample_num_matches=0
# run_eval \
#     "sift_pycolmap+roma" "roma" \
#     model.extractor.name=extractors.sift \
#     model.extractor.backend=pycolmap_cuda \
#     model.extractor.max_num_keypoints=4096 \
#     model.extractor.detection_threshold=0.0066667 \
#     model.extractor.rootsift=true \
#     model.extractor.nms_radius=0 \
#     model.extractor.first_octave=-1 \
#     model.extractor.num_octaves=4 \
#     model.extractor.init_blur=1.0 \
#     model.extractor.force_num_keypoints=False \
#     model.matcher.sample_num_matches=0
run_eval \
    "sift_py_cudasift+roma" "roma" \
    model.extractor.name=extractors.sift \
    model.extractor.backend=py_cudasift \
    model.extractor.max_num_keypoints=4096 \
    model.extractor.detection_threshold=0.00000000001 \
    model.extractor.rootsift=true \
    model.extractor.nms_radius=0 \
    model.extractor.first_octave=-1 \
    model.extractor.num_octaves=4 \
    model.extractor.init_blur=1.0 \
    model.extractor.force_num_keypoints=false \
    model.extractor.filter_kpts_with_wrapper=false \
    model.matcher.sample_num_matches=0 

# run_eval "aliked+nn" "aliked+NN"
# run_eval "aliked+lightglue_official" "aliked+lightglue-official"
# run_eval \
#     "aliked+roma" "roma" \
#     model.extractor.name=extractors.aliked \
#     model.extractor.max_num_keypoints=2048 \
#     model.extractor.detection_threshold=0.0 \
#     model.matcher.sample_num_matches=0

# run_eval "disk+nn" "disk+NN"
# run_eval "disk+lightglue_official" "disk+lightglue-official"
# run_eval \
#     "disk+roma" "roma" \
#     model.extractor.name=extractors.disk_kornia \
#     model.extractor.max_num_keypoints=2048 \
#     model.extractor.detection_threshold=0.0 \
#     model.matcher.sample_num_matches=0

# run_eval "roma_only" "roma"

#!/usr/bin/env bash
set -euo pipefail

run_eval() {
    local tag="$1"
    shift 1
    echo "==> ${tag}"
    python -m gluefactory.eval.endopatches1800 \
        --conf eval_endopatches1800 \
        --tag "${tag}" \
        "${BASE_DATA_ARGS[@]}" \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")
BASE_DATA_ARGS=(
    data.read_dataset_from_disk=true
    data.saved_dataset_dir=endopatches1800
)

sp_official_common=(
    model.extractor.name=gluefactory_nonfree.superpoint
    model.extractor.max_num_keypoints=1024
    model.extractor.detection_threshold=0
    model.extractor.nms_radius=0
)

sift_pycolmap_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=pycolmap_cuda
    model.extractor.max_num_keypoints=1024
    model.extractor.detection_threshold=0.00000000001
    model.extractor.rootsift=true
    model.extractor.nms_radius=0
    model.extractor.first_octave=-1
    model.extractor.num_octaves=4
    model.extractor.init_blur=1.0
    model.extractor.force_num_keypoints=false
    model.extractor.trainable=false
    model.extractor.filter_kpts_with_wrapper=false
    model.extractor.filter_with_scale_weighting=true
    model.extractor.extractor_channel=grayscale
    model.extractor.filter_with_lowest_scale=false
    model.extractor.random_topk=false
)

sift_cudasift_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=py_cudasift
    model.extractor.max_num_keypoints=1024
    model.extractor.detection_threshold=0.00000000001
    model.extractor.rootsift=true
    model.extractor.nms_radius=0
    model.extractor.first_octave=-1
    model.extractor.num_octaves=4
    model.extractor.init_blur=1.0
    model.extractor.force_num_keypoints=false
    model.extractor.trainable=false
    model.extractor.filter_kpts_with_wrapper=false
    model.extractor.filter_with_scale_weighting=true
    model.extractor.extractor_channel=grayscale
    model.extractor.filter_with_lowest_scale=false
    model.extractor.random_topk=false
)

aliked_common=(
    model.extractor.name=extractors.aliked
    model.extractor.max_num_keypoints=1024
    model.extractor.detection_threshold=0.0
)

disk_common=(
    model.extractor.name=extractors.disk_kornia
    model.extractor.max_num_keypoints=1024
    model.extractor.detection_threshold=0.0
)

lg_common=(
    model.matcher.depth_confidence=-1
    model.matcher.width_confidence=-1
    model.matcher.filter_threshold=0.1
)

lg_official_common=(
    model.matcher.name=matchers.lightglue_pretrained
)

roma_common=(
    model.matcher.name=roma
    model.matcher.sample_num_matches=0
    model.matcher.max_kp_error=3
    model.matcher.filter_threshold=0.05
)

#region Superpoint Official
run_eval "superpoint_official+nn" \
    "${sp_official_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True

run_eval "superpoint_official+nn_th" \
    "${sp_official_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7

run_eval "superpoint_official+nn_th_ratio" \
    "${sp_official_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "superpoint_official+lightglue_official" \
    "${sp_official_common[@]}" \
    "${lg_official_common[@]}" \
    model.matcher.features=superpoint \
    "${lg_common[@]}"

run_eval "superpoint_official+roma" \
    "${sp_official_common[@]}" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.output_hw=[518,672] \
    model.matcher.weights=indoor

run_eval "superpoint_official+lg_ENDO_HOMO" \
    "${sp_official_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=superpoint \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/09-sp_official+lg_ENDO_HOMO/checkpoint_best.tar
#endregion

#region SIFT_colmap
run_eval "sift_pycolmap+nn" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True

run_eval "sift_pycolmap+nn_th" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7

run_eval "sift_pycolmap+nn_th_ratio" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "00-sift_colmap+lg_official" \
    "${sift_pycolmap_common[@]}" \
    "${lg_official_common[@]}" \
    model.matcher.features=sift \
    "${lg_common[@]}"

run_eval "01-sift_colmap+lg_OP_HOMO" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/01-py_colmap+lg_OP_HOMO/checkpoint_best.tar

run_eval "02-sift_colmap+lg_MD_3D" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/02-py_colmap+lg_MD_3D/checkpoint_best.tar

run_eval "sift_colmap+04_lg_MD_3D_cudasift"
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar

run_eval "sift_colmap+03_lg_OP_HOMO_cudasift" \
    "${sift_pycolmap_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/03-py_cudasift+lg_OP_HOMO/checkpoint_best.tar

run_eval "sift_pycolmap+roma" \
    "${sift_pycolmap_common[@]}" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.output_hw=[518,672] \
    model.matcher.weights=indoor
#endregion

#region SIFT_cudasift
run_eval "sift_cudasift+nn" \
    "${sift_cudasift_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True

run_eval "sift_cudasift+nn_th" \
    "${sift_cudasift_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7

run_eval "sift_cudasift+nn_th_ratio" \
    "${sift_cudasift_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "00-sift_cudasift+lg_official" \
    "${sift_cudasift_common[@]}" \
    "${lg_official_common[@]}" \
    model.matcher.features=sift \
    "${lg_common[@]}"

run_eval "03-sift_cudasift+lg_OP_HOMO" \
    "${sift_cudasift_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/03-py_cudasift+lg_OP_HOMO/checkpoint_best.tar

run_eval "sift_cudasift+roma" \
    "${sift_cudasift_common[@]}" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.output_hw=[518,672] \
    model.matcher.weights=indoor

run_eval "05-sift_cudasift+lg_ENDO_HOMO" \
    "${sift_cudasift_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/05-sift_cudasift+lg_ENDO_HOMO/checkpoint_best.tar

run_eval "04-sift_cudasift+lg_MD_3D"
    "${sift_cudasift_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar

run_eval "sift_cudasift+02_lg_MD_3D_pycolmap"
    "${sift_cudasift_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${lg_common[@]}" \
    checkpoint=/workspace/data/training_outputs/02-py_colmap+lg_MD_3D/checkpoint_best.tar

#endregion

#region ALIKED-n16
run_eval "aliked+nn" \
    "${aliked_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True

run_eval "aliked+nn_th" \
    "${aliked_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7

run_eval "aliked+nn_th_ratio" \
    "${aliked_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "aliked+lightglue_official" \
    "${aliked_common[@]}" \
    "${lg_official_common[@]}" \
    model.matcher.features=aliked \
    "${lg_common[@]}"

run_eval "aliked+roma" \
    "${aliked_common[@]}" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.output_hw=[518,672] \
    model.matcher.weights=indoor
#endregion

#region DISK
run_eval "disk+nn" \
    "${disk_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True

run_eval "disk+nn_th" \
    "${disk_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7

run_eval "disk+nn_th_ratio" \
    "${disk_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "disk+lightglue_official" \
    "${disk_common[@]}" \
    "${lg_official_common[@]}" \
    model.matcher.features=disk \
    "${lg_common[@]}"

run_eval "disk+roma" \
    "${disk_common[@]}" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.weights=indoor
#endregion

#region ROMA
run_eval "roma_indoor" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.sample_num_matches=1024 \
    model.matcher.weights=indoor

run_eval "roma_outdoor" \
    "${roma_common[@]}" \
    model.matcher.internal_hw=[518,518] \
    model.matcher.output_hw=[518,672] \
    model.matcher.sample_num_matches=1024 \
    model.matcher.weights=outdoor
#endregion

#region Report generation
python tools/report_summaries.py --benchmark endopatches1800 --format plain --sort-by H_error_ransac_mAA --descending
python tools/report_summaries.py --benchmark endopatches1800 --format csv --sort-by H_error_ransac_mAA --descending
python tools/report_summaries.py --benchmark endopatches1800 --format md --sort-by H_error_ransac_mAA --descending
#endregion

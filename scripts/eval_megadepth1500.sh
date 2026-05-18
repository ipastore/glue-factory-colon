#!/usr/bin/env bash
set -euo pipefail

run_eval() {
    local tag="$1"
    shift 1
    echo "==> ${tag}"
    python -m gluefactory.eval.megadepth1500 \
        --conf eval_megadepth1500 \
        --tag "${tag}" \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

SP_2k_common=(
    model.extractor.name=gluefactory_nonfree.superpoint
    model.extractor.max_num_keypoints=2048
    model.extractor.detection_threshold=0
    model.extractor.nms_radius=0
)

SiftGPU_4k_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=pycolmap_cuda
    model.extractor.max_num_keypoints=4096
    model.extractor.detection_threshold=0.0066667
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

CudaSift_4k_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=py_cudasift
    model.extractor.max_num_keypoints=4096
    model.extractor.detection_threshold=0.01
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

SiftGPU_2k_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=pycolmap_cuda
    model.extractor.max_num_keypoints=2048
    model.extractor.detection_threshold=0.0066667
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

CudaSift_2k_common=(
    model.extractor.name=extractors.sift
    model.extractor.backend=py_cudasift
    model.extractor.max_num_keypoints=2048
    model.extractor.detection_threshold=0.01
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

SP_4k_common=(
    model.extractor.name=gluefactory_nonfree.superpoint
    model.extractor.max_num_keypoints=4096
    model.extractor.detection_threshold=0
    model.extractor.nms_radius=0
)

ALIKED_2k_common=(
    model.extractor.name=extractors.aliked
    model.extractor.max_num_keypoints=2048
    model.extractor.detection_threshold=0.0
)

ALIKED_4k_common=(
    model.extractor.name=extractors.aliked
    model.extractor.max_num_keypoints=4096
    model.extractor.detection_threshold=0.0
)

DISK_2k_common=(
    model.extractor.name=extractors.disk_kornia
    model.extractor.max_num_keypoints=2048
    model.extractor.detection_threshold=0.0
)

DISK_4k_common=(
    model.extractor.name=extractors.disk_kornia
    model.extractor.max_num_keypoints=4096
    model.extractor.detection_threshold=0.0
)

LG_common=(
    model.matcher.depth_confidence=-1
    model.matcher.width_confidence=-1
    model.matcher.filter_threshold=0.1
)

LG_official_common=(
    model.matcher.name=matchers.lightglue_pretrained
)

RoMa_common=(
    model.matcher.name=roma
    model.matcher.sample_num_matches=0
    model.matcher.max_kp_error=3
    model.matcher.filter_threshold=0.05
)

#region SuperPoint Official
run_eval "SuperPoint_2k+NN" \
    "${SP_2k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7
run_eval "SuperPoint_4k+NN" \
    "${SP_4k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "SuperPoint_2k+LG-SP" \
    "${SP_2k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=superpoint \
    "${LG_common[@]}"
run_eval "SuperPoint_4k+LG-SP" \
    "${SP_4k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=superpoint \
    "${LG_common[@]}"

run_eval "SuperPoint_2k+RoMa" \
    "${SP_2k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
run_eval "SuperPoint_4k+RoMa" \
    "${SP_4k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
#endregion

#region SiftGPU
run_eval "SiftGPU_2k+NN" \
    "${SiftGPU_2k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7
run_eval "SiftGPU_4k+NN" \
    "${SiftGPU_4k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "SiftGPU_2k+LG-SiftGPU_official" \
    "${SiftGPU_2k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=sift \
    "${LG_common[@]}"
run_eval "SiftGPU_4k+LG-SiftGPU_official" \
    "${SiftGPU_4k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=sift \
    "${LG_common[@]}"


run_eval "SiftGPU_2k+LG-CudaSift_ours" \
    "${SiftGPU_2k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar
run_eval "SiftGPU_4k+LG-CudaSift_ours" \
    "${SiftGPU_4k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar


run_eval "SiftGPU_2k+LG-colon_ours" \
    "${SiftGPU_2k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/11-cudasift+lg_ENDO_ROMA_ft_04_pos_neg_ign_specular_mask/checkpoint_best.tar
run_eval "SiftGPU_4k+LG-colon_ours" \
    "${SiftGPU_4k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/11-cudasift+lg_ENDO_ROMA_ft_04_pos_neg_ign_specular_mask/checkpoint_best.tar

run_eval "SiftGPU_2k+RoMa" \
    "${SiftGPU_2k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
run_eval "SiftGPU_4k+RoMa" \
    "${SiftGPU_4k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
#endregion

#region CudaSift
run_eval "CudaSift_2k+NN" \
    "${CudaSift_2k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7
run_eval "CudaSift_4k+NN" \
    "${CudaSift_4k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "CudaSift_2k+LG-SiftGPU_official" \
    "${CudaSift_2k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=sift \
    "${LG_common[@]}"
run_eval "CudaSift_4k+LG-SiftGPU_official" \
    "${CudaSift_4k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=sift \
    "${LG_common[@]}"

run_eval "CudaSift_2k+LG-CudaSift_ours" \
    "${CudaSift_2k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar
run_eval "CudaSift_4k+LG-CudaSift_ours" \
    "${CudaSift_4k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/04-py_cudasift+lg_MD_3D/checkpoint_best.tar

run_eval "CudaSift_2k+LG-colon_ours" \
    "${CudaSift_2k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/11-cudasift+lg_ENDO_ROMA_ft_04_pos_neg_ign_specular_mask/checkpoint_best.tar
run_eval "CudaSift_4k+LG-colon_ours" \
    "${CudaSift_4k_common[@]}" \
    model.matcher.name=matchers.lightglue \
    model.matcher.features=sift \
    "${LG_common[@]}" \
    checkpoint=/workspace/data/training_outputs/11-cudasift+lg_ENDO_ROMA_ft_04_pos_neg_ign_specular_mask/checkpoint_best.tar

run_eval "CudaSift_2k+RoMa" \
    "${CudaSift_2k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
run_eval "CudaSift_4k+RoMa" \
    "${CudaSift_4k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
#endregion

#region ALIKED-n16
run_eval "ALIKED_2k+NN" \
    "${ALIKED_2k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7
run_eval "ALIKED_4k+NN" \
    "${ALIKED_4k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "ALIKED_2k+LG-ALIKED" \
    "${ALIKED_2k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=aliked \
    "${LG_common[@]}"
run_eval "ALIKED_4k+LG-ALIKED" \
    "${ALIKED_4k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=aliked \
    "${LG_common[@]}"

run_eval "ALIKED_2k+RoMa" \
    "${ALIKED_2k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
run_eval "ALIKED_4k+RoMa" \
    "${ALIKED_4k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
#endregion

#region DISK
run_eval "DISK_2k+NN" \
    "${DISK_2k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7
run_eval "DISK_4k+NN" \
    "${DISK_4k_common[@]}" \
    model.matcher.name=nearest_neighbor_matcher \
    model.matcher.mutual_check=True \
    model.matcher.distance_thresh=0.7 \
    model.matcher.ratio_thresh=0.7

run_eval "DISK_2k+LG-DISK" \
    "${DISK_2k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=disk \
    "${LG_common[@]}"
run_eval "DISK_4k+LG-DISK" \
    "${DISK_4k_common[@]}" \
    "${LG_official_common[@]}" \
    model.matcher.features=disk \
    "${LG_common[@]}"

run_eval "DISK_2k+RoMa" \
    "${DISK_2k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
run_eval "DISK_4k+RoMa" \
    "${DISK_4k_common[@]}" \
    "${RoMa_common[@]}" \
    model.matcher.weights=outdoor
#endregion

#region RoMa
run_eval "RoMa_2k" \
    "${RoMa_common[@]}" \
    model.matcher.internal_hw=[630,630] \
    model.matcher.sample_num_matches=2048 \
    model.matcher.weights=outdoor
run_eval "RoMa_4k" \
    "${RoMa_common[@]}" \
    model.matcher.internal_hw=[630,630] \
    model.matcher.sample_num_matches=4096 \
    model.matcher.weights=outdoor
#endregion

#region Report generation
python tools/report_summaries.py --benchmark megadepth1500 --format plain --sort-by rel_pose_error_mAA --descending
python tools/report_summaries.py --benchmark megadepth1500 --format csv --sort-by rel_pose_error_mAA --descending
python tools/report_summaries.py --benchmark megadepth1500 --format md --sort-by rel_pose_error_mAA --descending
#endregion

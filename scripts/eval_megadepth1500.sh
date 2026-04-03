#!/bin/bash

#region main function
run_eval() {
    local tag="$1"
    shift 1
    echo "==> ${tag}"
    python -m gluefactory.eval.megadepth1500 --conf eval_megadepth1500 --tag "${tag}" "$@" "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")
#endregion

#region Superpoint Official
################## Superpoint Official #################################
## NN-BI
run_eval "superpoint_official+nn" \
        model.extractor.name=gluefactory_nonfree.superpoint \
        model.extractor.max_num_keypoints=2048 \
        model.extractor.detection_threshold=0 \
        model.extractor.nms_radius=0 \
        model.matcher.name=nearest_neighbor_matcher

## NN-BI-TH
run_eval "superpoint_official+nn_th" \
        model.extractor.name=gluefactory_nonfree.superpoint \
        model.extractor.max_num_keypoints=2048 \
        model.extractor.detection_threshold=0 \
        model.extractor.nms_radius=0 \
        model.matcher.name=nearest_neighbor_matcher \
        model.matcher.filter_threshold=0.7

## NN-BI-TH-RATIO
run_eval "superpoint_official+nn_th_ratio" \
        model.extractor.name=gluefactory_nonfree.superpoint \
        model.extractor.max_num_keypoints=2048 \
        model.extractor.detection_threshold=0 \
        model.extractor.nms_radius=0 \
        model.matcher.name=nearest_neighbor_matcher \
        model.matcher.filter_threshold=0.7 \
        model.matcher.ratio_test_threshold=0.7

## LG - Official
run_eval "superpoint_official+lightglue_official" \
        model.extractor.name=gluefactory_nonfree.superpoint \
        model.extractor.max_num_keypoints=2048 \
        model.extractor.detection_threshold=0 \
        model.extractor.nms_radius=0 \
        model.matcher.name=matchers.lightglue_pretrained \
        model.matcher.features=superpoint \
        model.matcher.depth_confidence=-1 \
        model.matcher.width_confidence=-1 \
        model.matcher.filter_threshold=0.1 \
## ROMA
run_eval \
    "superpoint_official+roma" \
    model.extractor.name=gluefactory_nonfree.superpoint \
    model.extractor.max_num_keypoints=2048 \
    model.extractor.detection_threshold=0.0 \
    model.extractor.nms_radius=0 \
    model.matcher.name=roma \
    model.matcher.sample_num_matches=0 \
    model.matcher.max_kp_error=3 \
    model.matcher_filter_threshold=0.05 \
    model.matcher.weights=indoor

run_eval "superpoint_official+lg_ENDO_HOMO" \
        model.extractor.name=gluefactory_nonfree.superpoint \
        model.extractor.max_num_keypoints=2048 \
        model.extractor.detection_threshold=0 \
        model.extractor.nms_radius=0 \
        model.matcher.name=matchers.lightglue \
        model.matcher.features=superpoint \
        model.matcher.depth_confidence=-1 \
        model.matcher.width_confidence=-1 \
        model.matcher.filter_threshold=0.1 \
        checkpoint=/workspace/data/training_outputs/09-sp_official+lg_ENDO_HOMO/checkpoint_best.tar
# ################## Superpoint Official #################################
# #endregion

# #region SIFT_colmap
# ################## SIFT_colmap #################################
# ## NN-BI
# run_eval "sift_pycolmap+nn" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher

# ## NN-BI-TH
# run_eval "sift_pycolmap+nn_th" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7

# ## NN-BI-TH-RATIO
# run_eval "sift_pycolmap+nn_th_ratio" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7 \
#       model.matcher.ratio_test_threshold=0.7

# ## 00-LG-Official
# run_eval "00-sift_colmap+lg_official" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=matchers.lightglue_pretrained \
#       model.matcher.features=sift \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1

# ## 01-LG-OP_HOMO
# run_eval "01-sift_colmap+lg_OP_HOMO" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=matchers.lightglue \
#       model.matcher.features=sift \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \
#       checkpoint=/workspace/data/training_outputs/01-py_colmap+lg_OP_HOMO/checkpoint_best.tar

# ## 02-LG-MD_3D
# run_eval "02-sift_colmap+lg_MD_3D" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=pycolmap_cuda \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.0066667 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=matchers.lightglue \
#       model.matcher.features=sift \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \
#       checkpoint=/workspace/data/training_outputs/02-py_colmap+lg_MD_3D/checkpoint_best.tar

# ## ROMA
# run_eval \
#     "sift_pycolmap+roma" \
#     model.extractor.name=extractors.sift \
#     model.extractor.backend=pycolmap_cuda \
#     model.extractor.max_num_keypoints=4096 \
#     model.extractor.detection_threshold=0.0066667 \
#     model.extractor.rootsift=true \
#     model.extractor.nms_radius=0 \
#     model.extractor.first_octave=-1 \
#     model.extractor.num_octaves=4 \
#     model.extractor.init_blur=1.0 \
#     model.extractor.force_num_keypoints=false \
#     model.matcher.name=roma \
#     model.matcher.sample_num_matches=0 \
#     model.matcher.max_kp_error=3 \
#     model.matcher.filter_threshold=0.05 \
#     model.matcher.weights=indoor
# ################## SIFT_colmap #################################
# #endregion

# #region SIFT_cudasift
# ################## SIFT_cudasift #################################
# ## NN-BI
# run_eval "sift_cudasift+nn" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=py_cudasift \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.00000000001 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher

# ## NN-BI-TH
# run_eval "sift_cudasift+nn_th" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=py_cudasift \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.00000000001 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7

# ## NN-BI-TH-RATIO
# run_eval "sift_cudasift+nn_th_ratio" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=py_cudasift \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.00000000001 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7 \
#       model.matcher.ratio_test_threshold=0.7

# ## LG-Official
# run_eval "00-sift_cudasift+lg_official" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=py_cudasift \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.00000000001 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=matchers.lightglue_pretrained \
#       model.matcher.features=sift \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \


# ## 03-LG_OP_HOMO
# run_eval "03-sift_cudasift+lg_OP_HOMO" \
#       model.extractor.name=extractors.sift \
#       model.extractor.backend=py_cudasift \
#       model.extractor.max_num_keypoints=4096 \
#       model.extractor.detection_threshold=0.00000000001 \
#       model.extractor.rootsift=true \
#       model.extractor.nms_radius=0 \
#       model.extractor.first_octave=-1 \
#       model.extractor.num_octaves=4 \
#       model.extractor.init_blur=1.0 \
#       model.extractor.force_num_keypoints=false \
#       model.extractor.trainable=false \
#       model.extractor.filter_kpts_with_wrapper=false \
#       model.extractor.filter_with_scale_weighting=true \
#       model.extractor.extractor_channel=grayscale \
#       model.extractor.filter_with_lowest_scale=false \
#       model.extractor.random_topk=false \
#       model.matcher.name=matchers.lightglue \
#       model.matcher.features=sift \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \
#       checkpoint=/workspace/data/training_outputs/03-py_cudasift+lg_OP_HOMO/checkpoint_best.tar


# ## ROMA
# run_eval \
#     "sift_py_cudasift+roma" \
#     model.extractor.name=extractors.sift \
#     model.extractor.backend=py_cudasift \
#     model.extractor.max_num_keypoints=4096 \
#     model.extractor.detection_threshold=0.00000000001 \
#     model.extractor.rootsift=true \
#     model.extractor.nms_radius=0 \
#     model.extractor.first_octave=-1 \
#     model.extractor.num_octaves=4 \
#     model.extractor.init_blur=1.0 \
#     model.extractor.force_num_keypoints=false \
#     model.extractor.trainable=false \
#     model.extractor.filter_kpts_with_wrapper=false \
#     model.extractor.filter_with_scale_weighting=true \
#     model.extractor.extractor_channel=grayscale \
#     model.extractor.filter_with_lowest_scale=false \
#     model.extractor.random_topk=false \
#     model.matcher.name=roma \
#     model.matcher.sample_num_matches=0 \
#     model.matcher.max_kp_error=3 \
#     model.matcher.filter_threshold=0.05 \
#     model.matcher.weights=indoor

# ## 05-LG_ENDO_HOMO
# run_eval "05-sift_cudasift+lg_ENDO_HOMO" \
#     model.extractor.name=extractors.sift \
#     model.extractor.backend=py_cudasift \
#     model.extractor.max_num_keypoints=4096 \
#     model.extractor.detection_threshold=0.00000000001 \
#     model.extractor.rootsift=true \
#     model.extractor.nms_radius=0 \
#     model.extractor.first_octave=-1 \
#     model.extractor.num_octaves=4 \
#     model.extractor.init_blur=1.0 \
#     model.extractor.force_num_keypoints=false \
#     model.extractor.trainable=false \
#     model.extractor.filter_kpts_with_wrapper=false \
#     model.extractor.filter_with_scale_weighting=true \
#     model.extractor.extractor_channel=grayscale \
#     model.extractor.filter_with_lowest_scale=false \
#     model.extractor.random_topk=false \
#     model.matcher.name=matchers.lightglue \
#     model.matcher.features=sift \
#     model.matcher.depth_confidence=-1 \
#     model.matcher.width_confidence=-1 \
#     model.matcher.filter_threshold=0.1 \
#     checkpoint=/workspace/data/training_outputs/05-sift_cudasift+lg_ENDO_HOMO/checkpoint_best.tar
# # ################## SIFT_cudasift #################################
# # #endregion

# #region ALIKED-n16
# ################## ALIKED-n16 #################################
# # NN-BI
# run_eval "aliked+nn" \
#       model.extractor.name=extractors.aliked \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher

# # NN-BI-TH
# run_eval "aliked+nn_th" \
#       model.extractor.name=extractors.aliked \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7

# # NN-BI-TH-RATIO
# run_eval "aliked+nn_th_ratio" \
#       model.extractor.name=extractors.aliked \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7 \
#       model.matcher.ratio_test_threshold=0.7

# #LG
# run_eval "aliked+lightglue_official" \
#       model.extractor.name=extractors.aliked \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=matchers.lightglue_pretrained \
#       model.matcher.features=aliked \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \

# # ROMA
# run_eval \
#     "aliked+roma" \
#     model.extractor.name=extractors.aliked \
#     model.extractor.max_num_keypoints=2048 \
#     model.extractor.detection_threshold=0.0 \
#     model.matcher.name=roma \
#     model.matcher.sample_num_matches=0 \
#     model.matcher.max_kp_error=3 \
#     model.matcher.filter_threshold=0.05 \
#     model.matcher.weights=indoor
# ################## ALIKED-n16 #################################
# #endregion

# # #region DISK
# ################## DISK #################################
# # NN-BI
# run_eval "disk+nn" \
#       model.extractor.name=extractors.disk_kornia \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher

# # NN-BI-TH
# run_eval "disk+nn_th" \
#       model.extractor.name=extractors.disk_kornia \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7

# # NN-BI-TH-RATIO
# run_eval "disk+nn_th_ratio" \
#       model.extractor.name=extractors.disk_kornia \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=nearest_neighbor_matcher \
#       model.matcher.filter_threshold=0.7 \
#       model.matcher.ratio_test_threshold=0.7
# # LG
# run_eval "disk+lightglue_official" \
#       model.extractor.name=extractors.disk_kornia \
#       model.extractor.max_num_keypoints=2048 \
#       model.extractor.detection_threshold=0.0 \
#       model.matcher.name=matchers.lightglue_pretrained \
#       model.matcher.features=disk \
#       model.matcher.depth_confidence=-1 \
#       model.matcher.width_confidence=-1 \
#       model.matcher.filter_threshold=0.1 \
# # Roma
# run_eval \
#     "disk+roma" \
#     model.extractor.name=extractors.disk_kornia \
#     model.extractor.max_num_keypoints=2048 \
#     model.extractor.detection_threshold=0.0 \
#     model.matcher.name=roma \
#     model.matcher.sample_num_matches=0 \
#     model.matcher.max_kp_error=3 \
#     model.matcher.filter_threshold=0.05 \
#     model.matcher.weights=indoor

# ################## DISK #################################
# #endregion

#region ROMA
################## ROMA #################################
# End-to-end ROMA
# Indoor
# run_eval "roma_indoor"  \
#       model.matcher.name=roma \
#       model.matcher.internal_hw=[630,630] \
#       model.matcher.output_hw=[1344,1344] \
#       model.matcher.sample_num_matches=5000 \
#       model.matcher.weights=indoor

# # Outdoor
# run_eval "roma_outdoor"  \
#       model.matcher.name=roma \
#       model.matcher.internal_hw=[630,630] \
#       model.matcher.output_hw=[1344,1344] \
#       model.matcher.sample_num_matches=5000 \
#       model.matcher.weights=outdoor
################## ROMA #################################
#endregion

#region Report generation
#### Run report generation ######
#txt
python tools/report_summaries.py  --benchmark megadepth1500  --format plain --sort-by rel_pose_error_mAA --descending
#csv
python tools/report_summaries.py  --benchmark megadepth1500  --format csv --sort-by rel_pose_error_mAA --descending
#md
python tools/report_summaries.py  --benchmark megadepth1500  --format md --sort-by rel_pose_error_mAA --descending
#endregion

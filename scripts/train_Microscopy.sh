# V1 can load pretrained model of VSS Block

MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset703_NeurIPSCell/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset703_NeurIPSCell/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="0"

# train
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 703 2d all -tr ${MAMBA_MODEL} &&

# predict
echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset703_NeurIPSCell/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 703 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_349.pth" &&

echo "Computing F1..."
python evaluation/compute_cell_metric.py \
    --gt_path "data/nnUNet_raw/Dataset703_NeurIPSCell/labelsVal-instance-mask" \
    -s "${PRED_OUTPUT_PATH}" \
    -o "${EVAL_METRIC_PATH}" \
    -n "${MAMBA_MODEL}_703_2d"  &&

echo "Done."
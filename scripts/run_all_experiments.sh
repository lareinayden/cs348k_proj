#!/bin/bash
set -e

echo "Running sparse text..."
python scripts/run_sd_text_batch.py \
  --modality sparse \
  --guidance-scales 3.0 5.0 7.5 10.0

echo "Running dense text..."
python scripts/run_sd_text_batch.py \
  --modality dense \
  --guidance-scales 3.0 5.0 7.5 10.0

echo "Running empty + canny (Config 3)..."
python scripts/run_controlnet_canny_batch.py \
  --modality empty \
  --guidance-scales 3.0 5.0 7.5 10.0 \
  --controlnet-scales 1.0

echo "Running empty + segmentation (Config 4)..."
python scripts/run_controlnet_seg_batch.py \
  --modality empty \
  --guidance-scales 3.0 5.0 7.5 10.0 \
  --controlnet-scales 1.0

echo "Running dense + canny (Config 5)..."
python scripts/run_controlnet_canny_batch.py \
  --modality dense \
  --guidance-scales 3.0 5.0 7.5 10.0 \
  --controlnet-scales 1.0

echo "Running dense + segmentation (Config 6)..."
python scripts/run_controlnet_seg_batch.py \
  --modality dense \
  --guidance-scales 3.0 5.0 7.5 10.0 \
  --controlnet-scales 1.0

echo "Running ControlNet scale sweep (Configs 3–6, CFG 7.5)..."
python scripts/run_controlnet_canny_batch.py \
  --modality empty \
  --guidance-scales 7.5 \
  --controlnet-scales 0.5 1.0 1.5 2.0 \
  --controlnet-scale-sweep

python scripts/run_controlnet_seg_batch.py \
  --modality empty \
  --guidance-scales 7.5 \
  --controlnet-scales 0.5 1.0 1.5 2.0 \
  --controlnet-scale-sweep

python scripts/run_controlnet_canny_batch.py \
  --modality dense \
  --guidance-scales 7.5 \
  --controlnet-scales 0.5 1.0 1.5 2.0 \
  --controlnet-scale-sweep

python scripts/run_controlnet_seg_batch.py \
  --modality dense \
  --guidance-scales 7.5 \
  --controlnet-scales 0.5 1.0 1.5 2.0 \
  --controlnet-scale-sweep

echo "All experiments finished."

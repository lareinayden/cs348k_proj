#!/bin/bash
set -e

echo "Running sparse text..."
python scripts/run_sd_text_batch.py --modality sparse --guidance-scales 3.0 5.0 7.5 10.0

echo "Running dense text..."
python scripts/run_sd_text_batch.py --modality dense --guidance-scales 3.0 5.0 7.5 10.0

echo "Running empty + canny..."
python scripts/run_controlnet_canny_batch.py --modality empty --guidance-scales 3.0 5.0 7.5 10.0

echo "Running dense + canny..."
python scripts/run_controlnet_canny_batch.py --modality dense --guidance-scales 3.0 5.0 7.5 10.0

echo "Running empty + segmentation..."
python scripts/run_controlnet_seg_batch.py --modality empty --guidance-scales 3.0 5.0 7.5 10.0

echo "Running dense + segmentation..."
python scripts/run_controlnet_seg_batch.py --modality dense --guidance-scales 3.0 5.0 7.5 10.0

echo "All experiments finished."
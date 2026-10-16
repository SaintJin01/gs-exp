#!/usr/bin/env bash
set -euo pipefail

output_dir=/home/sj/work/gaussian-splatting/output/train_eval/sam_boxplot/all_cameras_near_outliers
mkdir -p "$output_dir"
exec >"$output_dir/interpolated_run.log" 2>&1
cd /home/sj/work/gs-exp
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/sj/anaconda3/envs/gaussian_splatting/lib/python3.7/site-packages/torch/lib
exec /home/sj/anaconda3/envs/gaussian_splatting/bin/python -u \
  save_q3_gaussians_all_cameras.py \
  -m /home/sj/work/gaussian-splatting/output/train_eval \
  --cameras_json /home/sj/work/gaussian-splatting/output/train_eval/sam_boxplot/cameras.json

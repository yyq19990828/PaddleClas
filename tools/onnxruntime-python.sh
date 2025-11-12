#!/usr/bin/bash

# python onnxruntime-python.py --model /path/to/your/model.onnx --data_dir /path/to/your/image/directory1 /path/to/your/image/directory2 --labels /path/to/your/labels.txt --val_file val.txt --eval --apply_sigmoid
# --data_dir dataset/VA/boden_policecar dataset/VA/boden_schoorbus dataset/VA/tyjt_1th_vehichle_attribute dataset/VA/tyjt_2th_vehichle_attribute\(20680_mainly_truck\) \


# 数据集路径变量，按需注释/取消注释
DATA_DIRS=( \
    dataset/VA/boden_1th \
    dataset/VA/boden_2th \
    dataset/VA/tyjt_2th_vehichle_attribute \
    dataset/VA/tyjt_1th_vehichle_attribute \
    dataset/VA/tyjt_3th_vehichle_attribute \
    dataset/VA/ruqi_1th \
    dataset/VA/suzhouwanglian_1th \
    dataset/VA/suzhouwanglian_2th
)

python tools/onnxruntime-python.py \
    --model ./output_inference/LCNet_224X224_0912/inference_dynamic.onnx \
    --data_dir "${DATA_DIRS[@]}" \
    --labels ./dataset/VA/label_old.txt \
    --val_file val_old.txt \
    --eval \
    --error_collection \
    --confusion_matrix \
    --save_ori_image
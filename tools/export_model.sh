获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

获取输入参数，默认为test
model_name=${1:-test}

python ./tools/export_model.py \
    -c /home/paddle_ws/PaddleClas/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml \
    -o Global.pretrained_model=./output_train/${model_name}/best_model \
    -o Global.save_inference_dir=./output_inference/${model_name}_${date_suffix} \
    -o Global.batch_infer=None

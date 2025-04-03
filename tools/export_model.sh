python ./tools/export_model.py \
    -c /home/paddle_ws/PaddleClas/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml \
    -o Global.pretrained_model=./output_train/LCNet_0402/best_model \
    -o Global.save_inference_dir=./output_inference/LCNet_0402 \
    -o Global.batch_infer=None

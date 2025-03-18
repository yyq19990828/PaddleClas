python ./tools/export_model.py \
    -c /home/paddle_ws/PaddleClas/ppcls/configs/PULC/vehicle_attribute/ResNet_CSRA.yaml \
    -o Global.pretrained_model=./output/CSRA/best_model \
    -o Global.save_inference_dir=./output_inference/CSRA_batch1 \
    -o Global.batch_infer=1 \
    -o global.image_shape="[3, 385, 512]"
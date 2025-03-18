python3 tools/infer.py \
    -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml \
    -o Global.pretrained_model=output/tyjt2/best_model/model.pdparams \
    -o Arch.name=ResNet101_vd \

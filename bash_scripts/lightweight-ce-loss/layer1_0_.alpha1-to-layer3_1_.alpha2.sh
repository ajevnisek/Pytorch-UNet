mkdir -p /storage/jevnisek/layer2layer-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2
python train_layer_to_layer_lightweight_unet.py --classes 1 --batch-size 128 --config-file /storage/jevnisek/layer2layer-lightweight/configs/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2.yaml --dir_checkpoint /storage/jevnisek/layer2layer-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2
mkdir -p /storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1
python train_layer_to_layer_lightweight_unet.py --classes 1 --unet-type SuperDuperLightweightUNetLayer2Layer --batch-size 128 --config-file /storage/jevnisek/layer2layer-super-duper-lightweight/configs/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1.yaml --dir_checkpoint /storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1
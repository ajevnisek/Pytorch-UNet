runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer10alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer1_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer10alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer11alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer1_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer11alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer1_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer20alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer2_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer20alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer2_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer21alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer2_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer21alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer2_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer30alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer3_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer30alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer3_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer31alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer3_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer31alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer3_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer40alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer4_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer40alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer4_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer41alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer4_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-l1-loss-layer10alpha1-to-layer41alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/l1-loss/layer1_0_.alpha1-to-layer4_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer10alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer10alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer11alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer11alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer20alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer20alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer21alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer21alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer30alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer30alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer31alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer31alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer40alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer40alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer41alpha1 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5
runai submit --name amir-layer2layer-ce-loss-layer10alpha1-to-layer41alpha2 -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5
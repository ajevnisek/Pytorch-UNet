runai submit --name amir-layer2layer-super-lightweight-0 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-1 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-2 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-3 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-4 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-5 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-6 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-7 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-8 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-9 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-10 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-11 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-12 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-13 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-14 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-lightweight-15 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha2.sh --pvc=storage:/storage --large-shm
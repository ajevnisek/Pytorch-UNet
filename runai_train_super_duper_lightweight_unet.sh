runai submit --name amir-layer2layer-super-duper-lightweight-0 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-1 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-2 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-3 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-4 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-5 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-6 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-7 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer2_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-8 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-9 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-10 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-11 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer3_1_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-12 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-13 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_0_.alpha2.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-14 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha1.sh --pvc=storage:/storage --large-shm
sleep 5 
runai submit --name amir-layer2layer-super-duper-lightweight-15 -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer4_1_.alpha2.sh --pvc=storage:/storage --large-shm
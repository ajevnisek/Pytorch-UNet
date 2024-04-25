# Docker instructions
1. Build the docker:
```shell
docker build -t unet-channels:v6 -f docker/v6_finetune_classifier_after_layer2layer_replacement_with_super_lightweight/Dockerfile .
docker tag unet-channels:v6 ajevnisek/unet-channels:v6
docker push ajevnisek/unet-channels:v6
```
2. Then run the docker:
3. 
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/say_hello.sh  -it unet-channels:v6
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars_finetune.sh  -e RUN_SCRIPT=/local_code/bash_scripts/short_local_finetune.sh -v /mnt/data/temp_cache/:/unets/trained/models/  -it unet-channels:v6```

On runai:
```shell
runai submit --name amir-layer2layer-lightweight-finetune-debug -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-lightweight/configs/finetune/export_vars_finetune_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-lightweight/configs/finetune/short_remote_finetune_lightweight.sh --pvc=storage:/storage --large-shm
runai submit --name amir-layer2layer-lightweight-finetune-all -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-lightweight/configs/finetune/export_vars_finetune_all_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-lightweight/configs/finetune/short_remote_finetune_lightweight.sh --pvc=storage:/storage --large-shm

runai submit --name amir-layer2layer-super-lightweight-finetune-debug -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/configs/finetune/export_vars_finetune_super_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/configs/finetune/short_remote_finetune_super_lightweight.sh --pvc=storage:/storage --large-shm
runai submit --name amir-layer2layer-super-lightweight-finetune-all -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/configs/finetune/export_vars_finetune_all_super_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/configs/finetune/short_remote_finetune_super_lightweight.sh --pvc=storage:/storage --large-shm

runai submit --name amir-layer2layer-super-duper-lightweight-finetune-debug -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/configs/finetune/export_vars_finetune_super_duper_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/configs/finetune/short_remote_finetune_super_duper_lightweight.sh --pvc=storage:/storage --large-shm
runai submit --name amir-layer2layer-super-duper-lightweight-finetune-all -g 1.0 -i ajevnisek/unet-channels:v6 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/configs/finetune/export_vars_finetune_all_super_duper_lightweight.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/configs/finetune/short_remote_finetune_super_duper_lightweight.sh --pvc=storage:/storage --large-shm
```


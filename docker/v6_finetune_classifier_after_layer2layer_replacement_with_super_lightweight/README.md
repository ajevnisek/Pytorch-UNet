# Docker instructions
1. Build the docker:
```shell
docker build -t unet-channels:v3 -f docker/v3_finetune_classifier_after_layer2layer_replacement/Dockerfile .
docker tag unet-channels:v3 ajevnisek/unet-channels:v3
docker push ajevnisek/unet-channels:v3
```
2. Then run the docker:
3. 
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/say_hello.sh  -it unet-channels:v3
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars_finetune.sh  -e RUN_SCRIPT=/local_code/bash_scripts/short_local_finetune.sh -v /mnt/data/temp_cache/:/unets/trained/models/  -it unet-channels:v3```
On runai:
```shell
runai submit --name amir-layer2layer-finetune-debug -g 1.0 -i ajevnisek/unet-channels:v3 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/configs/finetune/export_vars_finetune.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/configs/finetune/short_remote_finetune.sh --pvc=storage:/storage --large-shm
```


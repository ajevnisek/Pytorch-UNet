# Docker instructions
1. Build the docker:
```shell
docker build -t unet-channels:v5 -f docker/v5_layer2layer_super_and_super_duper_lightweight_unet/Dockerfile .
docker tag unet-channels:v5 ajevnisek/unet-channels:v5
docker push ajevnisek/unet-channels:v5
```
2. Then run the docker:
3. 
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/say_hello.sh  -it unet-channels:v5
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/short_local_run_super_lightweight_unet.sh  -it unet-channels:v5
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/short_local_run_super_duper_lightweight_unet.sh  -it unet-channels:v5

```
On runai:
```shell
runai submit --name amir-layer2layer-super-lightweight-debug -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
runai submit --name amir-layer2layer-super-duper-lightweight-debug -g 1.0 -i ajevnisek/unet-channels:v5 -e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer-super-duper-lightweight/ce-loss/layer1_0_.alpha1-to-layer1_0_.alpha2.sh --pvc=storage:/storage --large-shm
```


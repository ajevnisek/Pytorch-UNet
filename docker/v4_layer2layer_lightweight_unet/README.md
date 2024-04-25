# Docker instructions
1. Build the docker:
```shell
docker build -t unet-channels:v2 -f docker/v2_layer2layer/Dockerfile .
docker tag unet-channels:v2 ajevnisek/unet-channels:v2
docker push ajevnisek/unet-channels:v2
```
2. Then run the docker:
3. 
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/say_hello.sh  -it unet-channels:v2
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/bash_scripts/export_vars.sh  -e RUN_SCRIPT=/local_code/bash_scripts/short_local_run.sh  -it unet-channels:v2

docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training_change_specific_layer/exports/local_config_file.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training_change_specific_layer/ae_relu_alpha_training_change_specific_layers_runner.sh  -it unet-channels:v1
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/exports/local_relu_autoencoder_alpha_training_slow_update_low_freq.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/ae_relu_alpha_training_runner.sh  -it ajevnisek/unet-channels:v1 
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/exports/local_config_file.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/ae_relu_alpha_training_runner.sh  -it ajevnisek/unet-channels:v1 /bin/bash
#docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/exports/snlr_relu_ae_runner_15000_local.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_alpha_training/snl_relu_ae_runner.sh  -it ajevnisek/unet-channels:v1

```
On runai:
```shell
runai submit --name amir-layer2layer-debug -g 1.0 -i ajevnisek/unet-channels:v2 -e EXPORT_SCRIPT=/jevnisek/layer2layer/bash_scripts/export_vars.sh -e RUN_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/short_remote_run.sh --pvc=storage:/storage --large-shm
```


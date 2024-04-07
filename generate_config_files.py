import os
import yaml

from classifier_utils import LAYERNAMES
d = {'epochs': 10}
os.makedirs(os.path.join('configs', 'l1-loss',), exist_ok=True)
os.makedirs(os.path.join('configs', 'ce-loss',), exist_ok=True)
os.makedirs(os.path.join('bash_scripts', 'l1-loss',), exist_ok=True)
os.makedirs(os.path.join('bash_scripts', 'ce-loss',), exist_ok=True)
runai_commands = []

for layer_name in LAYERNAMES:
    config_name = f"{LAYERNAMES[0].replace('[', '_').replace(']', '_')}-to-{layer_name.replace('[', '_').replace(']', '_')}.yaml"
    config_path = os.path.join('configs', 'l1-loss', config_name)
    d['layername_in'] = LAYERNAMES[0]
    d['layername_out'] = layer_name
    d['criterion'] = 'l1-loss'
    with open(config_path, 'w') as f:
        yaml.dump(d, f)
    path_to_bash_script = os.path.join('bash_scripts', d['criterion'], config_name.replace('yaml', 'sh'))
    with open(path_to_bash_script, 'w') as f:
        dir_checkpoint = f"/storage/jevnisek/layer2layer/{d['criterion']}/{config_name.replace('.yaml', '')}"

        bash_script = [f"mkdir -p {dir_checkpoint}",
            f"python train_layer_to_layer.py --classes 1 --batch-size 128 "
                       f"--config-file /storage/jevnisek/layer2layer/{config_path.replace('[', '_').replace(']', '_')} "
                       f"--dir_checkpoint {dir_checkpoint}"]
        f.write('\n'.join(bash_script))
    runai_command = (f"runai submit "
                     f"--name amir-layer2layer-{d['criterion']}-{config_name.replace('.yaml', '').replace('.', '_').replace('_', '')} "
                     f"-g 1.0 -i ajevnisek/unet-channels:v2 "
                     f"-e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh "
                     f"-e RUN_SCRIPT=/storage/jevnisek/layer2layer/{path_to_bash_script} --pvc=storage:/storage --large-shm")
    print(runai_command)
    runai_commands.append(runai_command)
    runai_commands.append('sleep 5')

for layer_name in LAYERNAMES:
    config_name = f"{LAYERNAMES[0].replace('[', '_').replace(']', '_')}-to-{layer_name.replace('[', '_').replace(']', '_')}.yaml"
    config_path = os.path.join('configs', 'ce-loss', config_name)
    d['layername_in'] = LAYERNAMES[0]
    d['layername_out'] = layer_name
    d['criterion'] = 'ce-loss'
    with open(config_path, 'w') as f:
        yaml.dump(d, f)
    path_to_bash_script = os.path.join('bash_scripts', d['criterion'], config_name.replace('yaml', 'sh'))
    with open(path_to_bash_script, 'w') as f:
        dir_checkpoint = f"/storage/jevnisek/layer2layer/{d['criterion']}/{config_name.replace('.yaml', '')}"

        bash_script = [f"mkdir -p {dir_checkpoint}",
                       f"python train_layer_to_layer.py --classes 1 --batch-size 128 "
                       f"--config-file /storage/jevnisek/layer2layer/{config_path.replace('[', '_').replace(']', '_')} "
                       f"--dir_checkpoint {dir_checkpoint}"]
        f.write('\n'.join(bash_script))
    runai_command = (f"runai submit "
                     f"--name amir-layer2layer-{d['criterion']}-{config_name.replace('.yaml', '').replace('.', '_').replace('_', '')} "
                     f"-g 1.0 -i ajevnisek/unet-channels:v2 "
                     f"-e EXPORT_SCRIPT=/storage/jevnisek/layer2layer/bash_scripts/export_vars.sh "
                     f"-e RUN_SCRIPT=/storage/jevnisek/layer2layer/{path_to_bash_script} --pvc=storage:/storage --large-shm")
    print(runai_command)
    runai_commands.append(runai_command)
    runai_commands.append('sleep 5')

with open('runai_commands.sh', 'w') as f:
    f.write('\n'.join(runai_commands))

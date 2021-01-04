# dmvde

# Generate data
- Adapt basepaths in `./datageneration/create_dataset.py` and `./datageneration/ALR_Praktikum_cfg.yaml`
- load `dataset_environment.blend` in blender and run the script (Note: `dataset_environment.blend` is on `.gitignore`)
- HDRI images can be downloaded from [HDRI Heaven](https://hdrihaven.com/)

# run dataloader in tensorflow
- python3 demo.py --image_path "data"  --dataloader "monoculer" --config_yaml "config_yaml.yml"""


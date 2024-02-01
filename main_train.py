''' training script of DECA
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    if cfg.cfg_file is None:
        cfg.cfg_file = 'full_config.yaml'
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    # logs\full_config.yaml
    print("cfg_file:", cfg.cfg_file)
    print("output_dir:", cfg.output_dir)

    with open(cfg.cfg_file, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("Source cfg_file:", cfg.cfg_file)
    print("Destination path:", os.path.join(cfg.output_dir, 'config.yaml'))

    # shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
    src = cfg.cfg_file
    dst = os.path.join(cfg.output_dir, 'config.yaml')
    if src != dst:
        shutil.copy(src, dst)
    else:
        print("Source and destination are the same, copy operation skipped.")

    
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    from decalib.deca import DECA
    from decalib.trainer import Trainer
    cfg.rasterizer_type = 'pytorch3d'
    deca = DECA(cfg)
    trainer = Trainer(model=deca, config=cfg)

    ## start train
    trainer.fit()

if __name__ == '__main__':
    from decalib.utils.config import parse_args
    cfg = parse_args()
    if cfg.cfg_file is not None:
        print(f"Using configuration file: {cfg.cfg_file}")
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
        print('exp_name:', exp_name)
        print('cfg_file:', cfg.cfg_file)
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml 
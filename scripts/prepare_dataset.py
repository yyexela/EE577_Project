#!/home/ee577/miniconda3/envs/ee577/bin/python

import os
from pathlib import Path
import sys
import torch

pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)

from src import *

# Load config file
config = global_config.config
config.device = 'cuda:1'
torch.manual_seed(config.seed)

preprocess_config = {
    'modality': ['T2', 'FLAIR', 'T1', 'T1GD'], 
    'image_type': 'autosegm',
    'window': (140, 172, 164),
    'pad_window': (70, 86, 86),
    'base_dim': (155, 240, 240),
    'downsample': True,
    'window_idx': ((0, 140), (39, 211), (44,208)),
    'down_factor': 0.5,
    'augments': ['base']
}

data_prep.convert_image_data_mod(**preprocess_config)

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
    'down_factor': 1.0,
    'augments': ['base'],
    'append_mask' : True
}

data_prep.convert_image_data_mod(**preprocess_config)

###############################
# Imports # Imports # Imports #
###############################

import os
from argparse import Namespace
from pathlib import Path

###################################
# Initialization # Initialization #
###################################

def init():
    """
    Function called once to initialize the Config class

    Note: SERIOUSLY! Call this only once every time you run something. Every time it's initialized it gets re-written from scratch.
    """
    global config
    config = Config()

###############################
# Config Class # Config Class #
###############################

class Config(Namespace):
    """
    Config class helps manage all the different directories and hyperparameters used in this repo. Uses a `Namespace` internally to allow accessing variables (ex. `a`) as `<Config>.a` or `<Config>['a']. Implements dict functions.

    Official docs as a reference: https://docs.python.org/3/reference/datamodel.html

    Initialize once to instantiate the global variable.
    """
    def __init__(self) -> None:
        # Locate parent directory
        self.top_dir = str(Path(__file__).parent.parent.absolute())

        # Package name
        self.package_name = "mypkg"

        # Initialize normal values by defualt
        self.init_normal()

        return None

    def init_normal(self) -> None:
        """
        Initialize normal configuration values

        Returns: `None`
        """

        ###################################################
        # Dataset File Structure # Dataset File Structure # 
        ###################################################

        # Directory structure
        self.dataset_dir = os.path.join(self.top_dir, "Datasets")
        self.checkpoint_dir = os.path.join(self.top_dir, "Checkpoints")
        self.pickle_dir = os.path.join(self.top_dir, "Pickles")
        self.image_dir = os.path.join(self.top_dir, "Images")

        self.yalefaces_dir = os.path.join(self.dataset_dir, "yalefaces")

        # Create those directories
        for newpath in [self.dataset_dir, self.checkpoint_dir, self.pickle_dir, self.image_dir]:
            if not os.path.exists(newpath):
                os.makedirs(newpath)

        self.device = 'cpu'

        return None

    ########################
    # Dict-based Functions #
    ########################

    # Source: https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
    # These have been modified to update the internal Namespace

    def __setitem__(self, key, item):
        vars(self)[key] = item

    def __getitem__(self, key):
        return vars(self)[key]

    def __repr__(self):
        return repr(vars(self))

    def __len__(self):
        return len(vars(self))

    def __delitem__(self, key):
        del vars(self)[key]

    def clear(self):
        return vars(self).clear()

    def copy(self):
        return vars(self).copy()

    def has_key(self, k):
        return k in vars(self)

    def update(self, *args, **kwargs):
        return vars(self).update(*args, **kwargs)

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def items(self):
        return vars(self).items()

    def pop(self, *args):
        return vars(self).pop(*args)

    def __iter__(self):
        return iter(vars(self))

    def __str__(self):
        return str(repr(vars(self)))
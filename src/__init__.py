__all__ = ['helpers', 'global_config', 'plot_data', 'models', 'datasets']

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from mypkg.src.global_config import init

init()

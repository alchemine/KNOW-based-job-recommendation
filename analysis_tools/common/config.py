"""Configuration module

Commonly used constant parameters are defined in capital letters.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


### Common parameters
RANDOM_STATE = 42


### Plot parameters
SHOW_PLOT      = True
FIGSIZE_UNIT   = 5
FIGSIZE        = (5*FIGSIZE_UNIT, 3*FIGSIZE_UNIT)
BINS           = 50
N_CLASSES_PLOT = 5
N_COLS         = 5
LEARNING_CURVE_N_SUBSETS_STEP = 5


### Model selection
TEST_SIZE = 0.2


### Model
DROPOUT_RATE = 0.2


### PATH
from os.path import join, dirname
class PATH:
    ROOT   = dirname(dirname(dirname(__file__)))
    INPUT  = join(ROOT, 'KNOW_data')
    TRAIN  = join(INPUT, 'train')
    TEST   = join(INPUT, 'test')
    OUTPUT = join(ROOT, 'output')
    RESULT = join(ROOT, 'result')

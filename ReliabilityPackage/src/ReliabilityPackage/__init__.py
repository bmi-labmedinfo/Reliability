import os
import random
import numpy as np
import torch
from . import constants
from ReliabilityPackage.ReliabilityClasses import *
from ReliabilityPackage.ReliabilityFunctions import *
from ReliabilityPackage.ReliabilityPrivateFunctions import *


def set_seed(seed=constants.SEED):
    # Set the seed environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set the python built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # Set the numpy pseudo-random generator at a fixed value
    np.random.seed(seed)
    # Set the torch seed at a fixed value
    torch.manual_seed(seed)


set_seed()

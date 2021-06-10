import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int] = None) -> int:
    """Function that sets seed for pseudo-random number generators  in:
    pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

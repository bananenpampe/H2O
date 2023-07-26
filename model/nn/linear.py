import torch
import equistore
from typing import List, Tuple, Dict, Union, Optional
from .utils import l_to_str
from .modules import EquistoreLazyTorchApply

class EquistoreLinearLazy(EquistoreLazyTorchApply):
    
    def __init__(self, n_out: int):
        super().__init__(torch.nn.Linear, n_out)

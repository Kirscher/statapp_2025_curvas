"""
Custom trainer for the statapp application.

This module provides a custom trainer for nnunet with stop early and custom network initialization.

I also added emoji and color formatting because why not :)
"""
import os
import sys
from datetime import datetime
from time import time, sleep
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from rich.text import Text

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_Statapp import nnUNetTrainer_Statapp
from statapp.utils.utils import setup_logging, info, pretty_print


class nnUNetTrainer_StatappDebug(nnUNetTrainer_Statapp):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from skimage import color
from skimage import io
from torch.optim import Adam


from dataclasses import dataclass
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import interpolate


from tqdm import tqdm
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from numpy import genfromtxt

class HandPosModel(nn.Module):
    def __init__(self) -> None:
        super(HandPosModel, self).__init__()

        
        self.layer1 = nn.Sequential(
            nn.Linear(42, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84,42),
            nn.ReLU(inplace = True),
            nn.Linear(42, 21),
            nn.ReLU(inplace = True),
            nn.Linear(21, 7),

            
        )
       

    def forward(self, z: torch.Tensor) -> torch.Tensor:     

        z = self.layer1(z.view(z.size(0), -1))
        return z

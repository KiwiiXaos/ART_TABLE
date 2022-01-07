import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
#from data import load, GetImage, Datoset, Dataset
#from network import *
#from skimage import color
#from skimage import io


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

from PIL import Image


class GenLineart(nn.Module):
    def __init__(self) -> None:
        super(GenLineart, self).__init__()

        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            

            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256,256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(48,48, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.Sigmoid(),

            

            
        )
       

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        print(z.shape)
     

        z = self.layer1(z)
        print(z.shape)
        '''
  
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer65(z)
        z = self.layer66(z)
        z = self.layer67(z)
        z = self.layer7(z)
        z = self.layer8(z)
        z = self.layer9(z)

        #z = self.final(z)
        '''


        return z
    

class Discrimator(nn.Module):
    def __init__(self) -> None:
        super(Discrimator, self).__init__()

        
        self.disc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 4, kernel_size=2, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(40, 20),
            nn.ELU(inplace=True),
            nn.Linear(20, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 5),
            nn.ELU(inplace=True),
            nn.Linear(5, 1),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        print(z.shape)
     

        z = self.disc(z)
        print(z.shape)
        '''
  
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer65(z)
        z = self.layer66(z)
        z = self.layer67(z)
        z = self.layer7(z)
        z = self.layer8(z)
        z = self.layer9(z)

        #z = self.final(z)
        '''

        return z





        
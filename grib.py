import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from network_g import *
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

def LoadEval(sketch):

  #sketch = Image.open("./testa.png").convert('L')
  print("who are you",type(sketch), sketch.shape)
  #sketch = Image.fromarray(sketch)
  #sketch = sketch.convert('L')
  print(sketch)
  sketch = np.array(sketch)[None, ...]
  while sketch.shape[2] % 8 != 0:
      sketch = np.delete(sketch, 0, 2)

  while sketch.shape[1] % 8 != 0:
          #print("touché", sketch.shape[1])
        sketch = np.delete(sketch, 0, 1)
  print("grib shap", sketch.shape)
  return sketch

def Grib(image):
  transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5), (0.5))
          ]) 


  model = GenLineart()
  model.load_state_dict(torch.load('./MODEL/beta_histo_6_0.pth',  map_location=torch.device('cpu')))

  model.eval()
  test = LoadEval(image)
  test = transform(test)

  eval = test.view(len(test[1]), len(test), len(test[0][0])) #.cuda()

  output = model(eval[None, ...])

  output = output * 255
  output = output.detach()
  output = output.cpu().numpy()
  '''
  test = test * 266
  #test = test.detach()
  test = test.cpu().numpy()


  test = np.squeeze(test)
  '''
  stock = np.stack((test,)*3, axis=-1).astype(np.uint8)

  wow = np.squeeze(output)
  stacked_img = np.stack((wow,)*3, axis=-1).astype(np.uint8)


  print("output shape > ", wow.shape)
  print("stock shape > ", test.shape )
  #im = Image.fromarray(output)

  output = np.stack((output,)*3, axis=-1).astype(np.uint8)
  #test = np.stack((test,)*3, axis=-1).astype(np.uint8)
  print("output shape > ", output.shape)
  print("test shape >", test.shape)
  #im2 = Image.fromarray(test)
  #im2 = im2.convert('RGBA')
  #im2.save("./ori.png")


  output = np.squeeze(output)
  output = [ [ row[ i ] for row in output ] for i in range( len( output[ 0 ] ) ) ]
  output = np.array(output)
  im = Image.fromarray(output)
  im = im.convert('RGBA')
  #im = im.rotate(-90)
  im.save("./gribed.png")

  print(output)
  # IM => Image PIL
  return im



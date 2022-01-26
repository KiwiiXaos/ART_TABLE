import tkinter
import cv2
from PIL import Image
from PIL import ImageFile
import time
from AR_script import *
import torch.nn.functional as F
from typing import Callable, List


#from imagePros import *
from paintstorch.network import Generator, Illustration2Vec

class Sample(NamedTuple):
    y: torch.Tensor  # (3, H,   W  ) Target Illustration
    x: torch.Tensor  # (4, H,   W  ) Composition Input
    h: torch.Tensor  # (4, H/4, W/4) Color Hints
    c: torch.Tensor  # (3, H,   W  ) Segmented Color Map

def apply(*objects: List, transform: Callable) -> List:
    return [transform(o) for o in objects]

class Parallel(torch.nn.Module):
     def __init__(self):
             super().__init__()
             self.module = Generator(32, bn=False)

class ImageProcessed:
    def __init__(self, image, map):
        self.image = image
        self.w = image.shape[0]
        self.h = image.shape[1]
        self.WIDTH  = 512
        self.HEIGHT = 512
        self.H_HEIGHT = 128
        self.H_WIDTH = 128
        # REMAP !!!
        print("process",self.image.shape)

        self.image = cv2.resize(self.image, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
        # (4 H W)
        self.image = self.image.transpose(2,0,1)
        print("process",self.image.shape)

        self.image = np.expand_dims(self.image, axis=0)

        self.out = self.test(self.image, False)
        #self.image = np.repeat(self.image, 3, axis=0)
        print("process",self.image.shape)
        arr =  np.full((1,1,self.WIDTH, self.HEIGHT), 1)
        print(arr.shape, self.image.shape)
        self.image = np.append(self.image,arr, axis=1)
        print("aaaa", self.image.shape)


        
        self.PIXELS = self.WIDTH * self.HEIGHT
        
        self.H_WIDTH  = 128
        self.H_HEIGHT = 128
        self.H_PIXELS = self.H_WIDTH * self.H_HEIGHT
        self.paintorch = self.Process_Paintorch()
        print(type(self.image))
        print(self.image.shape)
 
        #self.h = np.full(( 1, self.H_WIDTH, self.H_HEIGHT), 0)
        #self.x = np.append(self.x, np.full((1,1,x.shape[2], x.shape[3]), 1), axis=1)

    def clone_unsqueeze(*objects: List) -> List:
        return apply(*objects, transform=lambda x: x.clone().unsqueeze(0))


    def test( self, sample: Sample, is_guide: bool,) -> np.ndarray:
        model = Parallel()
        ckpt = torch.load('MODEL/checkpoint_39.pth',map_location=torch.device('cpu'))
        fake = torch.zeros((1, 4, 512, 512))
        model.load_state_dict(ckpt)
        G = model.module


        F1 = torch.jit.trace(Illustration2Vec("./MODEL/i2v.pth").eval(), fake)

        f = F1(x)

        mask = x[:, -1].unsqueeze(1)
        fake, guide, residuals = G(x, h_, f)
        fake = x[:, :3] * (1 - mask) + fake.clip(-1, 1) * mask
        
        if is_guide:
            guide = c #c * (1 - mask) + C(guide, residuals).clip(-1, 1) * mask
        else:
            guide = c

        x, h, c, y, fake, guide = squeeze_permute_cpu_np(x, h, c, y, fake, guide)
        x, h, c = x[..., :3], h[..., :3], c[..., :3]
        
        img = (np.hstack([x, h, c, guide, y, fake]) * 0.5) + 0.5
        img = (img * 255).astype(np.uint8)
        return img




        

    def Process_Paintorch(self):
        x = (torch.from_numpy(self.image)/255 - 0.5)*2
        print(x)
        h = cv2.imread('illustration.png')

        print("hhhhh", h.shape)
        h = h.transpose(2,0,1)

        h = np.expand_dims(h, axis=0)
        arr =  np.full((1,1,self.H_WIDTH, self.H_HEIGHT), 1)
        h = np.append(h,arr, axis=1)




        h = torch.full((1,4,self.H_WIDTH, self.H_WIDTH),1)        

        print("essayy", {x.shape, h.shape}, "Identity")

        model = Parallel()
        ckpt = torch.load('MODEL/checkpoint_39.pth',map_location=torch.device('cpu'))
        fake = torch.zeros((1, 4, 512, 512))


        F1 = torch.jit.trace(Illustration2Vec("./MODEL/i2v.pth").eval(), fake)


        model.load_state_dict(ckpt)
        G = model.module
        x = torch.tensor(x, dtype=torch.float)
        f = F1(x)
        print("f222", f.shape)
        h = torch.tensor(h, dtype=torch.float)

        outp, bla , blo = G(x,h,f)
        print(outp)

        outp = outp * 255
        stock = np.squeeze(outp.detach().numpy()).astype(np.uint8)
        stock = np.stack((stock), axis=-1).astype(np.uint8)
        print("outp", stock.shape)

        cv2.imwrite('./test.png', stock)
        print("b",stock.shape)
        cv2.imwrite('./test.png', stock)
        






img = cv2.imread('testa3.png')
h = cv2.imread('illustration.png')
print(img.shape)
ImageProcessed(img,h)

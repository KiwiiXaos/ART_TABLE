from network import Generator, Illustration2Vec
from PIL import Image, ImageDraw
from tqdm import tqdm
from test_colormap import *
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import socket
from threading import Thread


class Parallel(torch.nn.Module):
     def __init__(self):
             super().__init__()
             self.module = Generator(32, bn=False)
def InitPT():
    model = Parallel()
    ckpt = torch.load('MODEL/checkpoint_39.pth',map_location=torch.device('cpu'))
    #fake = torch.zeros((1, 4, 512, 512))
    model.load_state_dict(ckpt)
    x = torch.zeros((2, 4, 512, 512))

    G = model.module
    PC = torch.jit.load("./MODEL/paintschainer.ts",map_location=torch.device('cpu'))
    F1 = torch.jit.trace(Illustration2Vec("./MODEL/i2v.pth").eval(), x)
    return G, PC, F1



def Paintorch_Improve(x,m,G,PC,F1, w, hw ):
    
    #x = torch.zeros((2, 4, 512, 512))
    print('elp',type(x))
    


    #samples = [ './wow/waw3']
    with torch.no_grad():
            print('shapee', x.shape)


            normalizedImg = np.zeros((512, 512))
            Color()
       
            h = Image.open( "./hnew.png")
            h = np.array(h.convert("RGBA").resize((512, 512))) / 255
            #m = np.array(m.convert("L").resize((512, 512))) / 255
            print("trum", m.shape)

            h[:, :, :3] = (h[:, :, :3] - 0.5) / 0.5

            h = torch.from_numpy(h).float().permute(2, 0, 1)
            h = F.interpolate(h.unsqueeze(0), size=(128, 128))

            y, *_ = G(x.unsqueeze(0), h, F1(x.unsqueeze(0)))
            y = y.squeeze(0).permute(1, 2, 0)
            y = y.cpu().detach().numpy()
            y = ((y * 0.5 + 0.5) * 255).astype(np.uint8)
            y = cv2.resize(y, (w,int(w*(297/210))), interpolation = cv2.INTER_AREA)
            print('shap',y.shape)
            #y = Image.fromarray(y)
            #print("d", y.size, type(y))
            #y = y.resize((w, h))

            Image.fromarray(y).save("y_paintstorch2.png")

            return y

def Paintorch_Gen(G,PC,F1):
        
    


    #ckpt = torch.load(args.model)

    #G = nn.DataParallel(Generator(args.features, bn=args.bn))
    #G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)
    #G = G.module.eval().cuda()

    x = torch.zeros((2, 4, 512, 512))


    #samples = [ './wow/waw3']
    with torch.no_grad():

        

            # ==== TORCH
            x = Image.open( "./model.png")
            w, hw = x.size
            #f = np.array(x)
            #w = f.shape[0]
            #hw = f.shape[1]
            

            imgx = np.array(x.convert("RGB").resize((512, 512))) 
            print('TYTYP',type(imgx))
            normalizedImg = np.zeros((512, 512))
            normalizedImg = cv2.normalize(imgx,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite("dnor.png", normalizedImg)

            ret, imgx = cv2.threshold(normalizedImg,160,255,cv2.THRESH_BINARY)
            cv2.imwrite("data.png", imgx)
            x = imgx.copy()

            print("sahpe",imgx.shape)
            #
            pass1 = np.full(imgx.shape, 255, np.uint8)

            im_inv = cv2.bitwise_not(imgx)
            mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
            _, pass1, _, _ = cv2.floodFill(imgx, None, (1,1),  newVal=(0, 0))
            print('shapeshape',np.squeeze(pass1[:,:,:1], axis = 2).shape, type(pass1[0,0,0]))
            cv2.imwrite( "aled2.png", pass1)
            #cv2.imwrite("")





            h = Image.open( "./h.png")

            #x = np.array(x.convert("RGB").resize((512, 512))) / 255
            x = x / 255
            
            h = np.array(h.convert("RGBA").resize((512, 512))) / 255
            print(pass1[:,:,:1].shape)
            m =  np.squeeze(pass1[:,:,:1], axis = 2)/ 255 #
            #m = np.array(m.convert("L").resize((512, 512))) / 255
            print("trum", m.shape)

            x = np.concatenate([x, m[:, :, None]], axis=-1)

            x[:, :, :3] = (x[:, :, :3] - 0.5) / 0.5
            h[:, :, :3] = (h[:, :, :3] - 0.5) / 0.5

            x = torch.from_numpy(x).float().permute(2, 0, 1)
            print("briprevprev", type(x))

            h = torch.from_numpy(h).float().permute(2, 0, 1)
            h = F.interpolate(h.unsqueeze(0), size=(128, 128))

            y, *_ = G(x.unsqueeze(0), h, F1(x.unsqueeze(0)))
            y = y.squeeze(0).permute(1, 2, 0)
            y = y.cpu().detach().numpy()
            y = ((y * 0.5 + 0.5) * 255).astype(np.uint8)
            print("w", w, "h", int(w*(297/210)), "y", type(y))
            y = cv2.resize(y, (w,int(w*(297/210))), interpolation = cv2.INTER_AREA)
            print('shap',y.shape)
            #y = Image.fromarray(y)
            #print("d", y.size, type(y))
            #y = y.resize((w, h))

            Image.fromarray(y).save("y_paintstorch2.png")

            print("briprev", type(x))

            return y, x, m, w, hw

            # ==== CHAINER

def main():
    G,PC,F1 = InitPT()

    ServerSideSocket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM) 
    host = socket.gethostname()   

    port = 9999       
    ServerSideSocket.bind((host, port))                                  
    ThreadCount = 0
                        
    while True:         
        print("boucle")                                 
        ServerSideSocket.listen(5) 

        Client, address = ServerSideSocket.accept()
        print("new")
        print('Connected to: ' + address[0] + ':' + str(address[1]))
        Thread(target = multi_threaded_client, args =(Client,G,PC,F1)).start()

        #start_new_thread(multi_threaded_client, (Client, ))
        ThreadCount += 1
        print('Thread Number: ' + str(ThreadCount))
    ServerSideSocket.close()

def multi_threaded_client(connection, G,PC,F1):

    print("ellà")
    init = 0 

    #connection.send("Welcome to paintorch client :)".encode())
    while True:
        data = connection.recv(2048)
        if( data.decode() == 'work'):
            y,x,m, w, hw = Paintorch_Gen(G,PC,F1)
            print("wow")
        else:
            print("not wow")
            y,x,m, w, hw = Paintorch_Gen(G,PC,F1)


        if not data:
            pass
        else: connection.sendall("wawaw".encode())
    connection.close()
    '''

    # Main LOOP 
    #socket INITIALISATION
    # create a socket object
    serversocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM) 

    # get local machine name
    host = socket.gethostname()                           

    port = 9999                                           

    # bind to the port
    serversocket.bind((host, port))                                  

    # queue up to 5 requests
    serversocket.listen(5)                                           

    while True:
        # establish a connection
        clientsocket,addr = serversocket.accept() 
        response = clientsocket.recv(255)

        if response != "":
                print(response.decode())
                y, x = Paintorch_Gen(G,PC,F1)
        clientsocket.send("done!".encode('ascii'))
    '''


'''
if __name__ == "__main__":
    main()
'''
#G, PC, F1 = InitPT()
#y,x,m, w, hw = Paintorch_Gen(G,PC,F1)
#print("bri", type(x))
#y =  Paintorch_Improve(x,m,G,PC,F1, w, hw)
#main()
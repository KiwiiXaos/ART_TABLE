import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from AR_script import *
#from imagePros import *
from paintstorch.network import Generator
#from tensorflow import keras 
from tensorflow.keras.models import model_from_json, model_from_config

import torch
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import json
import tensorflow.io

# TENSORFLOW 

def load_graph(frozen_graph_filename, x, h):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
        #tensorflow_graph = tf.saved_model.load(frozen_graph_filename)
    #new_model = tf.keras.models.load_model(frozen_graph_filename)
    print("FFFFF")
    for op in tf.compat.v1.get_default_graph().get_operations():		        
        print(str(op.name))		    
        for n in tf.get_default_graph().as_graph_def().node:		        
            print("FUC",n.name)
    sess= tf.compat.v1.Session(graph=graph)
    result=sess.run({ "input": x, "hints": h }, "Identity")

    return graph

def load_2(path):
    with tf.Session() as sess:		    
        with gfile.FastGFile(path, 'rb') as f:		        
            graph_def = tf.GraphDef()		        
            graph_def.ParseFromString(f.read())		        
            sess.graph.as_default()		        
            g_in = tf.import_graph_def(graph_def)		    
            for op in tf.get_default_graph().get_operations():		        
                print(str(op.name))		    
                for n in tf.get_default_graph().as_graph_def().node:		        
                    print(n.name)
    with tf.Session() as sess:		    
        with gfile.FastGFile('slopemodel/slopemodel.pb', 'rb') as f:		        
            graph_def = tf.GraphDef()		        
            graph_def.ParseFromString(f.read())		        
            sess.graph.as_default()		        
            g_in = tf.import_graph_def(graph_def)		    
            tensor_output = sess.graph.get_tensor_by_name('import/dense_2/Sigmoid:0')		    
            tensor_input = sess.graph.get_tensor_by_name('import/dense_1_input:0')		    
            predictions = sess.run(tensor_output, {tensor_input:sample})		    
            print(predictions)
    
class ImageProcessed:
    def __init__(self, image):
        self.image = image
        self.w = image.shape[0]
        self.h = image.shape[1]
        self.WIDTH  = 512
        self.HEIGHT = 512
        # REMAP !!!
        self.image = cv2.resize(self.image, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
        # (4 H W)
        self.image = np.expand_dims(self.image, axis=0)
        self.image = np.repeat(self.image, 3, axis=0)
        self.image = np.insert(self.image, 3,np.full((self.WIDTH, self.HEIGHT), 1), axis=0)
        #self.image[0].append(np.full((self.h, self.w), 1))
        # (1 4 h w)
        #self.image = np.expand_dims(self.image, axis=0)


        
        self.PIXELS = self.WIDTH * self.HEIGHT
        
        self.H_WIDTH  = 128
        self.H_HEIGHT = 128
        self.H_PIXELS = self.H_WIDTH * self.H_HEIGHT
        self.paintorch = self.Paintorched()
        print(type(self.image))
        print(self.image.shape)

        self.h = np.full(( 1, self.H_WIDTH, self.H_HEIGHT), 0)
        self.Process_Paintorch()
        


    def Paintorched(self):
        print("aie")

        #return load_2('./paintstorch/model/model.pb')
        
        #x = tf.zeros([1, 4, self.HEIGHT, self.WIDTH])
        #h = tf.zeros([1, 4, self.H_HEIGHT, self.H_WIDTH])

        return 0 #paintmodel(self.image)
    def Process_Paintorch(self):
        x = tf.math.divide(tf.math.subtract(tf.convert_to_tensor(self.image, np.float32), 0.5), 0.5)
        h = tf.math.divide(tf.math.subtract(tf.convert_to_tensor(self.h, np.float32), 0.5), 0.5)
        print("essayy")
        load_graph('/Users/celine/CloudStation/ESILV/ANNEE_5/A5_Project/ART_TABLE_PYTHON/paintstorch/model/model.pb', x, h)

        test = self.paintorch.predict(x,h)
        print("test", test.shape)






#ret, frame = self.vid.get_frame()
image = cv2.imread("image.jpg")

#frame = cv.cvtColor(frame, cv2.COLOR_RGB2BGR)
model = ScanPicture(image)
print("etape_1")
#ScanImage = tkinter.Canvas(self.window, width = model.shape[0], height = model.sha
print("imagz format", type(model))

wow = ImageProcessed(model)
print("wow???")

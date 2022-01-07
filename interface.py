import cv2
from PIL import Image
import PIL.Image
import time
import mediapipe
from threading import Thread
import numpy as np
import pygame
import torch
import matplotlib

from grib import *
from AR_script import *
from hand_model import HandPosModel


ref = ["dessin", "posé","index","main fermée", "cercle","écartée","pouce" ]

# Hand class, deal with gesture label, coordinate calibration, and updating Pygame.
class Hands():
    def __init__(self,name, calib, num, myfont, screen):
        self.affichage = Element(name, screen, calib, True)
        self.Media_x = 0 # The Finger Index Coordinate from Mediapipe / The camera
        self.Media_y = 0
        self.num = num # The id of the hand  
        self.label = 0
        self.labeled_gesture = ref[self.label] # jsp
        self.screen = screen
        screen.blit(self.affichage.image, (100,100))

    # Update the Index coordinates and hand gesture label
    def H_update_video(self, id, results, hands, frame, model): 
        for handLandmarks in results.multi_hand_landmarks: # Supprimer la boucle et gérer avec des ids !
                    #drawingModule.draw_landmarks(frame, results.multi_hand_landmarks[0], handsModule.HAND_CONNECTIONS)
                    j = 0
                    i =0
                    tab = []
                    for dataPoint in results.multi_hand_landmarks[id].landmark:
                    
                        tab.append(dataPoint.x)
                        tab.append(dataPoint.y)

                        print((dataPoint.x) * frame.shape[1],(dataPoint.y )* frame.shape[0] )

                        if i == 12:
                            self.Media_x = (dataPoint.x) * frame.shape[1]
                            self.Media_y = (dataPoint.y) * frame.shape[0]
                            #print((self.Media_x, self.Media_y))


                            ## Calibration déplacée dans element.update

                            '''
                            print(dataPoint.x)
                            if(dataPoint.x < array_calib[2][0] and dataPoint.x > array_calib[3][0] and dataPoint.y < array_calib[2][1] and dataPoint.y > array_calib[3][1]):
                                #print("YAYAYAY")
                                self.Media_x = (dataPoint.x - array_calib[3][0] )/(array_calib[2][0]-array_calib[3][0])
                                self.Media_y = (dataPoint.y - array_calib[3][1] )/(array_calib[2][1] - array_calib[3][1])
                            '''

                    

                    inp = torch.from_numpy(np.array([tab]))
                    label = model(inp.float())
                    self.label = torch.argmax(label, 1)[0]
    
        return self


    def H_update_PyGame(self, myfont):

        print("SELFFF", self.Media_y)

        self.affichage.update_PyGame([self.Media_x, self.Media_y])

        label_1 = myfont.render('Default_1', False, (0,0,0)) # Update and blit  calibrated coordinates
        self.screen.blit(label_1, (self.affichage.coord[0], self.affichage.coord[1] + 10))

        # Rajouter les events.. Les labels à l'affichage etc etc..
        return self




# Element à afficher.

class Element():
    def __init__(self,name, screen, calib, hand):
        self.name = name
        self.image = pygame.image.load(name).convert_alpha()
        self.coord = [0,0] # Coordinates on pygame
        self.calib = calib # The calibration reference
        self.track = [0,0] # The Coordinates on camera
        self.screen = screen
        self.check = hand
    def update_PyGame(self,coord):
        self.track = coord
        x, y = self.screen.get_size()

        # Calibration !
        if(self.check == True):
            if(self.track[0] < self.calib[2][0] and self.track[0] > self.calib[3][0] and self.track[1] < self.calib[2][1] and self.track[1] > self.calib[3][1]):
                self.coord[0] = (self.track[0] - self.calib[3][0] )/(self.calib[2][0]-self.calib[3][0])
                self.coord[1] = (self.track[1] - self.calib[3][1] )/(self.calib[2][1] - self.calib[3][1])
                self.coord[0] = x - x*self.coord[0]
                self.coord[1] = y - y*self.coord[1]
            
            print(self.coord, x, y)
            self.screen.blit(self.image, (self.coord))
        else: self.screen.blit(self.image, (self.coord[0], self.coord[1])) # For debugging

        
        #print("debug : calibrated coordinates", x - x*self.coord[0], x - x*self.coord[1]) 
        
        return self




# Main LOOP 

def Video(liste_main, liste_art, array_calib):
    #PYGAME INITIALISATION

    clock = pygame.time.Clock()
    WIDTH, HEIGHT = 1920,1080
    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    x, y = screen.get_size()

    # Calibration checking : DEBUG
    init = 1
    done = False

    # Add two hands 
    liste_main.append(Hands("repere1.png",array_calib_pg,0, myfont, screen)) # Rajouter un True et False pour les labels
    liste_main.append(Hands("repere2.png",array_calib_pg,1, myfont, screen))
    
    liste_main = [item.H_update_PyGame(myfont) for item in liste_main]

    pygame.display.flip()


    # Calibration
    vid_init = 1

    calibration = pygame.image.load("calib2.png").convert_alpha()
    screen.blit(calibration, (0,0))
    activate_1 = 0
    # Calibration Reference.
    Ref = cv.imread("calib.png")
    Ref = cv.cvtColor(Ref, cv.COLOR_BGR2GRAY)
    
    # Mediapipe !
    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands

    # Start the Video ! 

    capture = cv2.VideoCapture(1)

    # Hand Gesture Model ! 
    # hand_100epochs.ckpt is trained on horizontal dataset, 

    model = HandPosModel()
    model.load_state_dict(torch.load('./vertical.ckpt',  map_location=torch.device('cpu'))) 
    model.eval()

    # Some time to initialise.
    time.sleep(5)
    
    # Calibration.
    while(vid_init == 0):
        ret, frame = capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vid_init = Calibration(Ref, frame, 10, capture)



    # Main Loop.
    
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        while not done:
            ret, frame = capture.read()

            # CALIBRATION LOOP
            if init == 0:
                if(array_calib[0] == False):
                    screen.fill((0, 0, 0))
                    screen.blit(calibration, (0,0))
                elif(array_calib[1] == False):
                    #print(x)
                    screen.fill((0,0,0))
                    # La ref fait 50 pixels
                    screen.blit(calibration, (x-100,y-100))
                    array_calib_pg[0] = True

                else:
                    array_calib_pg[1] = True

                    init= 1

            
            else:

                screen.fill((0, 0, 0))

                # App FPS 
                fps = myfont.render(str(int(clock.get_fps())), False, (255,0,255))
                screen.blit(fps,(100,100))

                # Check the Labels

                # If the label is index 
                if(liste_main[0].label == 6): 
                    # Debug Message 
                    houra = myfont.render("Scanner l'image", False, (255,255,255))
                    screen.blit(houra,(100,300))

                    # The gesture has to be hold for 5 seconds to be activated.
                    if(activate_1 == 0):                    
                        activ_delay = pygame.time.get_ticks()
                        activate_1 = 1

                    else:
                        if(pygame.time.get_ticks() - activ_delay > 5000): 
                            houra = myfont.render("activation", False, (255,255,255))
                            screen.blit(houra,(200,400))
                            activate_1 = 0


                            #Tracking(frame)
                            # Should initialise new thread.
                            image = ScanPicture(frame)

                            # Add the scanned image 
                            liste_art.append(Element("Model.png", screen, array_calib, False))
                            liste_art = [item.update_PyGame([100,100]) for item in liste_art]

                            #Model 
                            image = Grib(image)
                            art = pygame.image.load("gribed.png").convert_alpha()

                else:
                    activate_1 = 0
                
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                for event in pygame.event.get(): # To quit
                    if event.type == pygame.QUIT:
                        done = True

            # Update Pygame
            clock.tick(60)
            liste_art = [item.update_PyGame([100,100]) for item in liste_art]
            liste_main = [item.H_update_PyGame(myfont) for item in liste_main]

            pygame.display.flip()
        
           
            # Hand Label update and tracking
            if results.multi_hand_landmarks != None:
                liste_main = [item.H_update_video(results, hands, frame, model) for item in liste_main]

                '''
                j = 0


                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
                    #print(handLandmarks)

                    i =0
            
                    tab = []
                    # Hand Gesture Objectttt UPDATE ( faut initialiser avant ..)
                    j += 1

                    #print("tab. siz", len(tab))
                '''



    cv2.destroyAllWindows()
    capture.release()
    


def Extraction(quota):
    capture = cv.VideoCapture(1)
    Ref = cv.imread("calib2.png")


    Ref = cv.cvtColor(Ref, cv.COLOR_BGR2GRAY)

    while (True):
        print("hey")
        Extra(Ref, capture, quota)
        
    cv.destroyAllWindows()
    capture.release()

# Debug tracking
def Tracking(frame):
    # DEBUG
    frame = cv.imread("marker33_test.png")
    #Load the dictionary that was used to generate the markers.

    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()
    

    # Detect the markers in the image
    #print(frame.shape)
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    coord = markerCorners[0][0][2]
    #coord = 
    if(coord[0] < array_calib[2][0] and coord[0] > array_calib[3][0] and coord[1] < array_calib[2][1] and coord[1] > array_calib[3][1]):
                                art_track[0] = (coord[0] - array_calib[3][0] )/(array_calib[2][0]-array_calib[3][0])
                                art_track[1] = (coord[1] - array_calib[3][1] )/(array_calib[2][1] - array_calib[3][1])
    print('MISSION ACCOMPLIE')



def Calibration(Ref, Frame, quota, vid):

    # From 
    ret, fra = vid.read()

    orb = cv.ORB_create()
    ref_keypts, ref_desc = orb.detectAndCompute(Ref, None)
    fr_keypts, fr_desc = orb.detectAndCompute(fra, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    ## DEBUG 1

    img = fra #cv2.imread('scene.jpg',0)
    print("debug ? CAMERA")

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
    im = Image.fromarray(img2)
    im = im.convert('RGBA')
    im.save("calibob.png")
    #cv2.waitKey(0)
    ##
    img = Ref #cv2.imread('scene.jpg',0)
    print("debug REF ? CAMERA")

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
    im = Image.fromarray(img2)
    im = im.convert('RGBA')
    im.save("calibob2.png")
            
    #matcher = cv.BFMatcher()
    matches = bf.match(ref_desc, fr_desc)
    print("[CAMERA] Matches :", matches)


    # DEBUG
    '''

    final_img = cv.drawMatches(Ref, ref_keypts,
    Frame, fr_keypts, matches[:20],None)
    final_img = cv.resize(final_img, (1000,650))
    # Show the final image
    #cv.imshow("Matches", final_img)
    cv.imwrite("test.png", final_img)
    '''
    ### END DEBUG
    #print(matches)
    

    if len(matches) > 0:
       
        ref_pts = np.float32([ref_keypts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        fr_pts = np.float32([fr_keypts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
       
        if array_calib[0] == False:
            x_point_debut = (fr_pts[0,0,0] )#/2 #+ fr_pts[1,0,0]
            y_point_debut = (fr_pts[0,0,1] )#/2 #+ fr_pts[1,0,1]
            array_calib[2] = [x_point_debut, y_point_debut]
            array_calib[0] = True
            time.sleep(5)

            return 0
        elif array_calib[1] == False:
            x_point_fin = (fr_pts[0,0,0])#/#2#+ fr_pts[1,0,0]
            y_point_fin = (fr_pts[0,0,1])#+ fr_pts[1,0,1])/2
            array_calib[3] = [x_point_fin, y_point_fin]
            array_calib[1] =True
            return 0
        elif array_calib[0] ==True and array_calib[1] == True:
            return 1
    else: return 0
#def GetImage():
matplotlib.pyplot.switch_backend('Agg') 
print("ah")
hand_loc =[0,0,0,0,0,0] # x1 y1 x2 y2 pos1 pos2
operation = [0,0,0]
liste_main = []
liste_art = []
#array_calib =[False, False, 0, 0]
array_calib = [True, True, [1699.0, 813.0], [538.0, 186.0]]
art_track = [0,0]
array_calib_pg = [True, True, [1699.0, 813.0], [538.0, 186.0]]
#class ImageTake():
#Thread(target = Video).start()
Video(liste_main, liste_art,array_calib)
#Projector()
# Create a window and pass it to the Application object
App(tkinter.Tk(), "ART_TABLE")
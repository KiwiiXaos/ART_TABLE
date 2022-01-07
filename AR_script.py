import cv2 as cv
import time
import numpy as np
import imutils
import sys
from transform import *
from skimage.filters import threshold_local
from grib import *
from PIL import ImageOps





def Extraction(quota):
    capture = cv.VideoCapture(1)
    Ref = cv.imread("calib2.png")


    Ref = cv.cvtColor(Ref, cv.COLOR_BGR2GRAY)
    #Frame = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)
    ret, Frame = capture.read()


    while (True):
        ScanPicture(Frame, ratio=500)
        
    cv.destroyAllWindows()
    capture.release()

def Extra(Ref, vid, quota):
    #ret, Frame = vid.read()
    Frame = cv.imread("calib2.png")


    Frame = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    ref_keypts, ref_desc = orb.detectAndCompute(Ref, None)
    fr_keypts, fr_desc = orb.detectAndCompute(Frame, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  


    #matcher = cv.BFMatcher()
    matches = bf.match(ref_desc, fr_desc)

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

    if len(matches) > quota:
        print("?", fr_keypts[0].pt)
        print(Ref.shape)
            
        ref_pts = np.float32([ref_keypts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        fr_pts = np.float32([fr_keypts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        #print("hey",ref_pts[0].shape#.pt)
        # compute Homography
        M, mask = cv.findHomography(ref_pts, fr_pts, cv.RANSAC, 5.0)
        print(Ref)

        return matches
    else: return None




# Part of this code is based from pyimagesearch.
def ScanPicture(image, ratio = 500):
    #image = cv.imread("image.jpg")

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    adjusted = gray # = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    edged = cv.Canny(adjusted, 75, 200)

    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    print("wraped", warped.shape)

    warped = Image.fromarray(warped)

    warped = warped.rotate(180)
    warped = warped.resize(((int(warped.size[0]*1.5), int(warped.size[1]*1.5))), Image.ANTIALIAS)
    warped = np.array(warped)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped2 = ((np.full(warped.shape, 1) - warped).astype("uint8") )* 255
    warped = np.full(warped.shape, 1) - warped
    print("wraped", warped.shape)
    print(warped)
    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv.imwrite("./model.png", warped2)
    return warped2


def main() -> int:

    image = cv.imread("image.png")

    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    im = ScanPicture(image)
    im = Grib(im)

        
    


if __name__ == '__main__':
    sys.exit(main())  
    

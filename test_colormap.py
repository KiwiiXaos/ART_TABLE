import cv2
import numpy as np

def Color():
    Ref2 = cv2.imread("./model.png")

    # convert image to hsv colorspace

    hsv = cv2.cvtColor(Ref2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print('v', type(v), v, type(v[0]))
    print('s', type(s), s, np.amax(s), type(s[0,0]))

    s = s * 1.7
    s = s.astype(np.uint8)
    s[s > 255] = 255
    print('s', type(s), s, np.amax(s), type(s[0,0]))


    thresh1 = cv2.threshold(s, 45, 255, cv2.THRESH_BINARY)[1]
    thresh1 = 255 - thresh1 

    #hresh1 = s 
    # threshold value image and invert
    thresh2 = cv2.threshold(v, 120, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2

    # combine the two threshold images as a mask
    mask = cv2.add(thresh1,thresh2)
    mask = 255 - mask

    # use mask to remove lines in background of input
    hsv = hsv[...,1] * 1.2
    hsv[hsv > 255] = 255
    imghsv = cv2.merge([h,s,v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    result = imgrgb.copy()
    result[mask==0] = (255,255,255)
    result = cv2.resize(result, (512,512), interpolation = cv2.INTER_AREA)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    
    result = cv2.GaussianBlur(result,(9,9),sigmaX=100.0)


    alpha = result[:,:,3]
    alpha[np.all(result[:, :, 0:3] == (255, 255, 255), 2)] = 0
    alpha[np.all(result[:, :, 0:3] >= (110, 110, 110), 2)] = 0




    # display IN and OUT images
    
    #cv2.imshow('RESULT', result)

    # save output image
    cv2.imwrite('mask.png', result)
    cv2.imwrite('symbols_thresh1.png', thresh1)
    cv2.imwrite('symbols_thresh2.png', thresh2)
    cv2.imwrite('symbols_mask.png', mask)
    cv2.imwrite('hnew.png', result)

Color()
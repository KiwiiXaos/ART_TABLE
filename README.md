# ART_TABLE

https://we.tl/t-5LWPKH9V2U


https://we.tl/t-s7ZoGcPgtE

webtransfer https://we.tl/t-Nt0xE0SRpt

https://we.tl/t-xcaBFdPuqq

Art Table is an interactive AR platform to draw. The user is able to draw on a sheet of paper, the interface scans the sketch and generates a digital inked version.
<p align="center">

<img width="475" alt="Capture d’écran 2022-01-07 à 14 50 08" src="https://user-images.githubusercontent.com/43905857/148553241-51ded036-4d2a-4470-a169-608175d0fd30.png">
  </p>


## How to run the project : ##

```python
python3 interface.py 
```
Note that the project is coded to use an external webcam. ```capture = cv.VideoCapture(1)```

## Dependencies :
- Torch
- Pygame
- numpy
- mediapipe
- PIL

## How does it works :

Art Table Architecture is physically composed of a camera and a projector which provide respectively the input and output. The project is coded in Python and works with Pytorch, OpenCV and Pygame.

To use the AR Interface, we’re using Hand gesture recognition with Mediapipe and a classification full linear model. It has seven output in total. Currently, only the thumb up gesture is used to project the lineart.



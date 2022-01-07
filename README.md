# ART_TABLE

Art Table is an interactive AR platform to draw. The user is able to draw on a sheet of paper, the interface scans the sketch and generates a digital inked version.
<p align="center">

<img width="475" alt="Capture d’écran 2022-01-07 à 14 50 08" src="https://user-images.githubusercontent.com/43905857/148553241-51ded036-4d2a-4470-a169-608175d0fd30.png">
  </p>


## How to run the project : ##

```python
python3 interface.py 
```
Note that the project is coded to use an external webcam. ```capture = cv.VideoCapture(1)```

## How does it works :

Art Table Architecture is physically composed of a camera and a projector which provide respectively the input and output. The project is coded in Python and works with Pytorch, OpenCV and Pygame.

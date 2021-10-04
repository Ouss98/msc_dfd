import cv2
import os
import imutils
from imutils.video import FPS
import math

def croppedRect(roi, x1, x2, y1, y2 ):
    roi = roi[y1:y2, x1:x2]
    return roi

def face_detection(image, cascPath):

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    #Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) != 0:
        success = True
    else:
        success = False
    print(faces)
    face = faces[0]
    print(face)
    return (success, face)

def extract_faces(videoPath):
    # Check if directory exists, otherwise create it
    if not os.path.exists('./savedFrames'):
        os.mkdir('./savedFrames')

    haarCascade = './haarcascade_frontalface_default.xml'

    vs = cv2.VideoCapture(videoPath)

    # cv2.CAP_PROP_FRAME_COUNT
    frame_count = int(vs.get(7))
    frame_number = 0
    frame_rate = vs.get(5) # fps

    try:
        count = 0
        test_count = 0
        # Loop over frames from the video stream
        while(vs.isOpened()):
            frame_number = vs.get(1) # current frame number
            # Grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            ret, frame = vs.read()
            # print(np.shape(frame))
            # print(f'frame = {frame}')
            # print(frame[1].shape)

            if ret != True:
                break

            if frame_number % math.floor(frame_rate) == 0:
                test_count = test_count + 1

            if (frame_number % math.floor(frame_rate) == 0):
                # Resize the frame (so we can process it faster) and grab the
                # frame dimensions
                frame = imutils.resize(frame, width=500)

                # Grab the new bounding box coordinates of the object
                (success, box) = face_detection(frame, haarCascade)
                # Check to see if the tracking was a success
                # if success:
                (x, y, w, h) = [int(v) for v in box]
        
                # cv2.CAP_PROP_POS_FRAMES
                frame_number = int(vs.get(1))
        
                cv2.imwrite('./savedFrames/frame%d.jpg' % count, croppedRect(frame, x, x + w, y, y + h))
                count = count + 1
                
            print(f'test_count = {test_count}')
        vs.release()
        
    except AttributeError:
        print('Video read')


videoPath = './test_vid2.mp4'
extract_faces(videoPath)

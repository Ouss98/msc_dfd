import cv2
import os
import imutils
import math
from mtcnn import MTCNN
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def croppedRect(roi, x1, x2, y1, y2 ):
    roi = roi[y1:y2, x1:x2]
    return roi

def face_detection(image):
    detector = MTCNN()
    results = detector.detect_faces(image)

    return results

def extract_faces(videoPath):
    # Check if directory exists, otherwise create it
    if not os.path.exists('./savedFrames_mtcnn'):
        os.mkdir('./savedFrames_mtcnn')

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
                res = face_detection(frame)
                # Bounding box
                bounding_box = res[0]['box']
                # Confidence
                confidence = res[0]['confidence']

                if confidence > 0.95:
                    # Check to see if the tracking was a success
                    x, y, w, h = bounding_box
                    # rect = cv2.rectangle(frame, (x, y), (x + w, y + h),
                    #     (0, 255, 0), 2)
          
                    cv2.imwrite('./savedFrames_mtcnn/frame%d.jpg' % count, croppedRect(frame, x, x + w, y, y + h))
                    count += 1
                else:
                    print('Skipped a face...')
            
            print(f'test_count = {test_count}')
        vs.release()
        
    except AttributeError:
        print('Video read')


videoPath = './test_vid2.mp4'
extract_faces(videoPath)

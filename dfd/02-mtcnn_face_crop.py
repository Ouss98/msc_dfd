import cv2
import os
import imutils
import math
import json
from mtcnn import MTCNN
from pkg_resources import add_activation_listener
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time

start_time = time.time()

train_fake_path = '.\\z_split_dataset\\train\\fake\\'
train_real_path = '.\\z_split_dataset\\train\\real\\'
val_fake_path = '.\\z_split_dataset\\val\\fake\\'
val_real_path = '.\\z_split_dataset\\val\\real\\'
test_fake_path = '.\\train_sample_videos\\test_videos\\fake\\'
test_real_path = '.\\train_sample_videos\\test_videos\\real\\'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def croppedRect(roi, x1, x2, y1, y2 ):
    roi = roi[y1:y2, x1:x2]
    return roi

def face_detection(image):
    detector = MTCNN()
    results = detector.detect_faces(image)

    return results

def rescale_frame(frame):
    if frame.shape[1] < 300:
        scale_ratio = 2
    elif frame.shape[1] > 1900:
        scale_ratio = 0.33
    elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
        scale_ratio = 0.5
    else:
        scale_ratio = 1

    width = int(frame.shape[1] * scale_ratio)
    height = int(frame.shape[0] * scale_ratio)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    return frame

def padding(bounding_box, image):
    margin_x = bounding_box[2] * 0.3  # 30% as the margin
    margin_y = bounding_box[3] * 0.3  # 30% as the margin
    x1 = int(bounding_box[0] - margin_x)
    if x1 < 0:
        x1 = 0
    x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
    if x2 > image.shape[1]:
        x2 = image.shape[1]
    y1 = int(bounding_box[1] - margin_y)
    if y1 < 0:
        y1 = 0
    y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
    if y2 > image.shape[0]:
        y2 = image.shape[0]

    # print(x1, y1, x2, y2)
    return x1, y1, x2, y2

def extract_faces_and_save(videoPath, targetPath, filename):
    # Check if directory exists, otherwise create it
    if not os.path.exists(targetPath):
        print('Creating directory : ' + targetPath)
        os.makedirs(targetPath)

    vs = cv2.VideoCapture(videoPath)

    frame_number = 0
    frame_rate = vs.get(5) # fps

    count = 0
    # Loop over frames from the video stream
    while(vs.isOpened()):
        frame_number = vs.get(1) # current frame number
        # Grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        success, frame = vs.read()

        # Check to see if we have reached the end of the stream
        if success != True:
            break

        if (frame_number % math.floor(frame_rate) == 0):
            # Resize the frame (so we can process it faster) and grab the
            # frame dimensions

            res = face_detection(frame)
            
            for faces in res:
                # Bounding box
                bounding_box = faces['box']
                # Confidence
                confidence = faces['confidence']
                # print(f'confidence = {confidence}')

                if len(res) < 2 or confidence > 0.95:
                    # Check to see if the tracking was a success
                    x, y, w, h = padding(bounding_box, frame)

                    crop = croppedRect(frame, x, w, y, h)
                    new_filename = '{}-{:02d}.png'.format(os.path.join(targetPath, get_filename_only(filename)), count)
                    count = count + 1
                    cv2.imwrite(new_filename, crop)
                else:
                    print('Skipped a face...')
    vs.release()
    print("Done!")

def extract_test_faces_and_save(videoPath, targetPath, filename):
    # Check if directory exists, otherwise create it
    if not os.path.exists(targetPath):
        print('Creating directory : ' + targetPath)
        os.makedirs(targetPath)

    vs = cv2.VideoCapture(videoPath)

    frame_number = 0
    frame_rate = vs.get(5) # fps

    count = 0
    # Loop over frames from the video stream
    while(vs.isOpened()):
        frame_number = vs.get(1) # current frame number
        # Grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        success, frame = vs.read()

        # Check to see if we have reached the end of the stream
        if success != True:
            break

        if (frame_number % (3*math.floor(frame_rate)) == 0):

            res = face_detection(frame)
            
            for faces in res:
                # Bounding box
                bounding_box = faces['box']
                # Confidence
                confidence = faces['confidence']
                # print(f'confidence = {confidence}')

                if len(res) < 2 or confidence > 0.95:
                    # Check to see if the tracking was a success
                    x, y, w, h = padding(bounding_box, frame)

                    crop = croppedRect(frame, x, w, y, h)                       
                    new_filename = '{}-{:02d}.png'.format(os.path.join(targetPath, get_filename_only(filename)), count)
                    count = count + 1
                    cv2.imwrite(new_filename, crop)
                else:
                    print('Skipped a face...')
    vs.release()
    print("Done!")

train_fake_files = [f for f in os.listdir(train_fake_path) if os.path.isfile(os.path.join(train_fake_path, f))]
train_real_files = [f for f in os.listdir(train_real_path) if os.path.isfile(os.path.join(train_real_path, f))]
val_fake_files = [f for f in os.listdir(val_fake_path) if os.path.isfile(os.path.join(val_fake_path, f))]
val_real_files = [f for f in os.listdir(val_real_path) if os.path.isfile(os.path.join(val_real_path, f))]
test_fake_files = [f for f in os.listdir(test_fake_path) if os.path.isfile(os.path.join(test_fake_path, f))]
test_real_files = [f for f in os.listdir(test_real_path) if os.path.isfile(os.path.join(test_real_path, f))]

train_val_dict = {
    'train_f': {'path': train_fake_path, 'files': train_fake_files},
    'train_r': {'path': train_real_path, 'files': train_real_files},
    'val_f': {'path': val_fake_path, 'files': val_fake_files},
    'val_r': {'path': val_real_path, 'files': val_real_files}
}

test_dict = {
    'test_f': {'path': test_fake_path, 'files': test_fake_files},
    'test_r': {'path': test_real_path, 'files': test_real_files}
}

def create_dataset(dict, extract_func):
    for value in dict.values():
        print(value['files'])
        for file in value['files']:
            if (file.endswith(".mp4")):
                video_path = os.path.join(value['path'], file)
                extract_func(video_path, value['path'], file)
            else:
                continue

create_dataset(train_val_dict, extract_faces_and_save)
print('Train and Validation sets done...')
create_dataset(test_dict, extract_test_faces_and_save)
print('Test set done...\nReady to train!')

print("--- %s seconds ---" % (time.time() - start_time))
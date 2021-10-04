import json
import os
from distutils.file_util import copy_file
import shutil
import numpy as np
import splitfolders

base_path = '.\\train_sample_videos\\'
dataset_path = '.\\z_prepared_dataset\\'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

for filename in metadata.keys():
    print(filename)
    print(metadata[filename]['label'])
    tmp_path = os.path.join(base_path, filename)  
    if metadata[filename]['label'] == 'REAL':  
        print('Copying to :' + real_path)
        # Copy entire source tree ('tmp_path') to 'real_path'
        copy_file(tmp_path, real_path)
    elif metadata[filename]['label'] == 'FAKE':
        print('Copying to :' + fake_path)
        # Copy entire source tree ('tmp_path') to 'fake_path'
        copy_file(tmp_path, fake_path)
    else:
        print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real videos: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(fake_path) if os.path.isfile(os.path.join(fake_path, f))]
print('Total Number of Fake videos: ', len(all_fake_faces))

# Split into Train / Val folders
# With 80% / 20% ratio
splitfolders.ratio(dataset_path, output='z_split_dataset', seed=1377, ratio=(.8, .2))
print('Train / Val Split Done!')





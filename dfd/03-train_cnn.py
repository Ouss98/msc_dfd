import os
import pandas as pd
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

# Set to force CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if tf.test.gpu_device_name():
#    print('GPU found')
#else:
#    print("No GPU found")

dataset_path = '.\\z_split_dataset\\'

# tmp_debug_path = '.\\tmp_debug'
# print('Creating Directory: ' + tmp_debug_path)
# os.makedirs(tmp_debug_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, concatenate, LeakyReLU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel

# Preprocess data
input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = '.\\train_sample_videos\\test_videos\\'

train_datagen = ImageDataGenerator(
    rescale = 1/255,    # rescale the tensor values to [0, 1]
    rotation_range = 10, # random rotation in [0°, 10°]
    width_shift_range = 0.1, # 10% of total width
    height_shift_range = 0.1, # 10% of total width
    shear_range = 0.2, # Shear Intensity
    zoom_range = 0.1, # range of random zoom [0.9, 1.1]
    horizontal_flip = True, # randomly flip inputs horizontally
    fill_mode = 'nearest' # filled following aaaaaaaa|abcd|dddddddd
)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
    #save_to_dir = tmp_debug_path
)

val_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0, 1]
)

val_generator = val_datagen.flow_from_directory(
    directory = val_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
    #save_to_dir = tmp_debug_path
)

test_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0, 1]
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

# Create model
# '''
# EfficientNet
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

# Compile model
model.compile(optimizer = Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback to save best model
custom_callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model_effnet.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks
)
print(history.history)
# '''

'''
# Meso4
def meso4():
    x = Input(shape = (input_size, input_size, 3))
    
    x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    
    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    
    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation = 'sigmoid')(y)

    return KerasModel(inputs = x, outputs = y)

model2 = meso4()
model2.summary()

# Compile model
model2.compile(optimizer = Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback to save best model
custom_callbacks2 = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model_meso4.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
num_epochs = 20
history2 = model2.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks2
)
print(history2.history)
'''

'''
# RCNN
def rcnn():
    # inception
    inputs = Input(shape = (input_size, input_size, 3))

    layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(inputs)
    layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

    layer_2 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_2)

    mid_1 = concatenate([layer_1, layer_2], axis = 3)

    # Flatten the output and add Dense layers
    flat_1 = Flatten()(mid_1)

    y = Dense(150, activation='relu')(flat_1)
    y = Dense(150, activation='relu')(y)
    outputs = Dense(2048, activation='softmax')(y)
    
    # lstm
    inputs_lstm = Reshape((1, 2048))(outputs)

    input_shape = (1, 2048)
    lstm_out = 512

    x = LSTM(units=lstm_out, input_shape=input_shape, dropout=0.5, return_sequences=True)(inputs_lstm)
    x = Flatten()(x)
    outputs_lstm = Dense(1, activation='sigmoid')(x)

    return KerasModel(inputs = inputs, outputs = outputs_lstm)

model2 = rcnn()
model2.summary()

# Compile model
model2.compile(optimizer = Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback to save best model
custom_callbacks2 = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model_rcnn.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
num_epochs = 20
history2 = model2.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks2
)
print(history2.history)
'''

# Test the model
# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model_effnet.h5'))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose = 1
)

test_results = pd.DataFrame({
    'Filename': test_generator.filenames,
    'Prediction': preds.flatten(),
    'Class': np.where(preds.flatten()<0.5, 'Deepfake', 'Authentic')
})

print(test_results)

# Export results to .csv format
test_results.to_csv('effnet_test_results.csv')
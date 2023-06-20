## import the libraries
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf 
from keras import backend as K
import gc
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import splitfolders
import os

# Define the directories for the original dataset and the split dataset
ORIGINAL_DATA_DIR = 'data'  # Directory containing the original dataset
OUTPUT_DIR = 'split_dataset'  # Directory to store the split dataset

# Split the dataset into training and validation sets
def split_dataset():
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the split ratios (adjust as needed)
    split_ratio = (0.8, 0.2)  # Train:Validation

    # Split the dataset using splitfolders
    splitfolders.ratio(ORIGINAL_DATA_DIR, output=OUTPUT_DIR, seed=42, ratio=split_ratio, group_prefix=None)

# Call the split_dataset function before training the model
split_dataset()

# Initialization of the functions and directories, and adding the parameters in our project.
IMG_WIDTH, IMG_HEIGHT = (150, 150)

TRAIN_DATA_DIR = 'split_dataset/train'
VALIDATION_DATA_DIR = 'split_dataset/val'
NB_TRAIN_SAMPLES = 10
NB_VALIDATION_SAMPLES = 10
EPOCHS = 100
BATCH_SIZE = 10

# Create our model that will undergo training later
def build_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate here
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


## Train the model
def train_model(model):
    rotation_range = 20
    width_shift_range = 0.2
    height_shift_range = 0.2
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE
    )
    
    return model


# Save the trained model to a file
def save_model(model):
    model.save('saved_model.h5')


# Define the main function
def main():
    tf.keras.backend.clear_session()
    gc.collect()
    
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)


# Call the main function to start the training process
main()

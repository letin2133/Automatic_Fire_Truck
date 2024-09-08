import cv2
import numpy as np 
import os 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

TRAIN_DIR = 'DATASET/TRAIN'
TEST_DIR = 'DATASET/TEST'

x_train = []
x_test = []


dict = { 'NONFIRE_IMAGE': 0, 'FIRE_IMAGE' : 1,
        'TEST_FIRE': 1, 'TEST_NONFIRE' : 0}




def getData(data_dir, lst_data) :
    for whatever in os.listdir(data_dir) :
        whatever_path = os.path.join(data_dir,whatever)
        lst_filename_path = []
        for file_name in os.listdir(whatever_path) :
            file_name_path = os.path.join(whatever_path,file_name)
            label = file_name_path.split('\\')[1]
            img = np.array(Image.open(file_name_path))
            lst_filename_path.append((img, dict[label]))
        
        lst_data.extend(lst_filename_path)

    return lst_data

x_train = getData(TRAIN_DIR,x_train)
x_test = getData(TEST_DIR, x_test)


for i in range (5) :
     np.random.shuffle(x_train)


model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Changed softmax to sigmoid
])

model_cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_cnn.fit(np.array([x[0] for _,x in enumerate(x_train)]), np.array([x[1] for _,x in enumerate(x_train)]), epochs = 200)
model_cnn.export('saved_model')



converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()
with open('converted_model.tflite', 'wb') as f:
  f.write(tflite_model)

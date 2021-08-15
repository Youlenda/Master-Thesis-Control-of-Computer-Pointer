'''
18 shahrivar



{'0': 0, '1': 2, '2': 1, '3': 3}
'''

import os
import sys
import cv2
import time
import glob
import pyautogui
import numpy as np
from PIL import Image
from numpy import loadtxt
from swish import swish_act
from keras.models import Model 
from keras import backend as K
import matplotlib.pyplot as plt
import efficientnet.keras as enet
from keras.backend import sigmoid
from keras.models import load_model
from keras.layers import Activation
from pynput.mouse import Button, Controller
from utils import detector_utils as detector_utils
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects
from keras.layers import BatchNormalization, Dropout, Dense


class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

#______________________CLASSIFIER, Distance Models_____________________________
model = enet.EfficientNetB0(include_top=False, input_shape=(70, 70, 3),
                            pooling='avg', weights='efficientnet_b0.h5')
x = model.output
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('swish_act')(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('swish_act')(x)
x = Dense(128)(x)
x = Activation('swish_act')(x)
x = Dense(64)(x)
x = Activation('swish_act')(x)
x = Dense(32)(x)
x = Activation('swish_act')(x)
x = Dense(16)(x)
x = Activation('swish_act')(x)
x = Dense(8)(x)
x = Activation('swish_act')(x)

predictions = Dense(4, activation="softmax")(x)
model_final = Model(inputs = model.input, outputs = predictions)

model_final.load_weights('Final_Model_Weight_18shah_1.h5')

model_final.layers.pop()
model_distance = Model(model_final.input, model_final.layers[-17].output)

#loading vectors
fist = loadtxt('fist.csv',  delimiter=',')
palm = loadtxt('palm.csv',  delimiter=',')
left = loadtxt('left.csv',  delimiter=',')
righ = loadtxt('right.csv', delimiter=',')


#____________________________Some Useful Functions_____________________________
#Mouse
mouse = Controller()

#Camera
camera = cv2.VideoCapture(0)
def capture_image():
    return_value, frame = camera.read()
    return frame
#SSD Detector
detection_graph, sess = detector_utils.load_inference_graph()


#____________________________Some Useful Prameters_____________________________
#Offset
xmin_of, ymin_of = 10, 30
xmax_of, ymax_of = 10, 10

#Predict
command = 0
labels = [0, 0]
offset = -5

#Mouse
x = 0
y = 0
alfa = 0.5
save_x = []
save_y = []
(screenx,screeny) = (1440, 900)
(capturex,capturey) = (300,300)

mouse_ = False
text = None
font = cv2.FONT_HERSHEY_COMPLEX

print('> ====== Your cool mouse is ready.')
print('> ====== Please 🖐 to start.')  

#__________________________________MAIN BODY___________________________________
while True:
#    tic = time.time()
    #Take a photo.
    img = capture_image()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    
    #Detect a hand by SDD.
    boxes, scores = detector_utils.detect_objects(img, detection_graph, sess)

    for i in range(0, 1):
        if scores[0] < 0.15:
            calibration = 1
        elif scores[0] > 0.15:
            #Bounding Boxe Coordinate
            bbx = boxes[i] * 300
            ymin = int(bbx[0]) - ymin_of
            xmin = int(bbx[1]) - xmin_of
            ymax = int(bbx[2]) + ymax_of
            xmax = int(bbx[3]) + xmax_of
            
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 9, 100), 3)
            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0
            if ymax < 0:
                ymax = 0
            if xmax < 0:
                xmax = 0
            
            detections = ((xmin, ymin, xmax, ymax))
            
#            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           
            plt.imshow(img)
            plt.show()

            
                
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #Crop the photo (if there is a hand).
            img_crop = img[ymin:ymax, xmin:xmax]
            #Center of Bounding Box
            x = (xmax+xmin)/2
            y = (ymax+ymin)/2
            #Save the photo! (because there was some error)
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            cv2.imwrite('img_crop/' + str(33) + '.jpg', img_crop)
            address = '/Users/yalda/Desktop/Project/all parts/ssd/img_crop/33.jpg'
            Images = glob.glob(address)
#            try:
                #Load the photo and prepare it for models.
            img_crop = Image.open(Images[0]).convert("RGB")
            img_crop = img_crop.resize((70, 70))
                

                
            img_crop = np.array(img_crop)
            img_crop = np.array(img_crop, dtype='float32')
            img_crop = img_crop/255
            img_crop = img_crop.reshape((1, 70, 70, 3))
            #Use photo as models' input
            
            
            test_datagen = ImageDataGenerator()
            test_generator = test_datagen.flow(img_crop, shuffle=False)
            #Predict label of the photo
            predict_test = model_final.predict_generator(test_generator)
            predicted_label = np.argmax(predict_test)
            #Calculate distance of photo with classes
            img_fea = model_distance.predict_generator(test_generator)
            img_fea = img_fea.reshape(1280)
            
            dis_fist = np.linalg.norm(fist-img_fea)
            dis_palm = np.linalg.norm(palm-img_fea)
            dis_left = np.linalg.norm(left-img_fea)
            dis_righ = np.linalg.norm(righ-img_fea)
            
            dis = [dis_fist-1, dis_righ - 6, dis_palm, dis_left]
#            print(dis)
            
            if np.argmin(dis) == 0:
                if dis_fist > 34.9-offset:
                    command = 0
                else:
                    command = 1
            elif np.argmin(dis) == 2:
                if dis_palm > 34.3-offset:
                    command = 0
                else:
                    command = 1
            elif np.argmin(dis) == 3:
                if dis_left > 46-offset:
                    command = 0
                else:
                    command = 1
            elif np.argmin(dis) == 1:
                if dis_righ + 10 > 45.5-offset:
                    command = 0
                else:
                    command = 1

            
            if command == 1:
                labels.append(predicted_label)
                
#_____________________________________FIST_____________________________________
                if predicted_label == 0 and np.argmin(dis) == 0:
                    print('✊: Your cool mouse is off')

#_____________________________________PALM_____________________________________
                if predicted_label == 2 and np.argmin(dis) == 2:
                    print('🖐: You are in tracking mode.')
                        
                if predicted_label == 3 and np.argmin(dis) == 3:
                    print('👈: You clicked.')
                        
                if predicted_label == 1 and np.argmin(dis) == 1:
                    print('👉: You right-clicked.')
            else:
                print('OTHERS')


cv2.destroyAllWindows()
del(camera)
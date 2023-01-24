#this version exclusively using the tutorial here: https://data-flair.training/blogs/handwritten-character-recognition-neural-network/
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import PIL.ImageOps
from PIL import Image

AZ_MAP = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 

def load_existing_model(fn="convolutional_NN_for_AZ_recognition.keras"):
    return tf.keras.models.load_model(fn)

def go():
    train_x, train_y, test_x, test_y = get_data()
    print("converting single float values to categorical values for the convolutional neural network...")
    train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
    test_yOHE  = to_categorical(test_y,  num_classes=26, dtype='int')
    print("new shape of train labels: ", train_yOHE.shape, " --- test labels new shape: ", test_yOHE)
    model = define_and_compile_model()
    history = model.fit(train_x, train_yOHE, epochs=1, validation_data=(test_x, test_yOHE))
    return model
    
def test_letters(model):
    letters = ["B.png", "D.jpeg", "Q.png", "W.png"]
    for fn in letters:
        print("\n\n===========\n\nrunning for ", fn)
        x = handwritten_letter(fn)
        run_predictions(x, model)
        plt.imshow(x)
        plt.show()

def run_predictions(x, model):
    xmod = np.reshape(x, (1, 28, 28, 1))#for CNN input
    prediction_p = model.predict(xmod)[0]
    top_prediction_p_indices  = np.argsort(-prediction_p)[:3] #return the indices of the highest 3 values for probability P
    x = x.reshape((28,28))
    for i in range(3):
        print("i=", i)
        prediction_p_i = top_prediction_p_indices[i]
        #print("prediction_p_i.type=", type(prediction_p_i), "   prediction_p_i=", prediction_p_i)   
        predicted_letter = AZ_MAP[prediction_p_i]
        pct_possibility =  prediction_p[prediction_p_i] * 100.0
        pct_possibility = round(pct_possibility, )
        print(pct_possibility, "% chance the letter shown is a ", predicted_letter)


def handwritten_letter(fn):
    img_array = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img_array)
    img_pil = PIL.ImageOps.invert(img_pil)
    img28 = np.array(img_pil.resize((28,28), Image.ANTIALIAS))
    return img28
    
def define_and_compile_model():
    model = Sequential(
            [
                Conv2D( filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1) ),
                MaxPool2D( pool_size=(2,2), strides=2 ),
                Conv2D( filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                MaxPool2D( pool_size=(2,2), strides=2),
                Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'),
                MaxPool2D( pool_size=(2,2), strides=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(26, activation='softmax')
            ]
        )
    model.compile( optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'] )
    return model  

def get_data():
    X = np.load("X_ocr_set.npy")
    y = np.load("y_ocr_set.npy")
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.2)
    train_x = np.reshape(train_x, (train_x.shape[0],28,28))
    test_x = np.reshape(test_x, (test_x.shape[0],28,28))
    return (train_x, train_y, test_x, test_y)

def display_some_letters(train_x):
    shuff = shuffle(train_x[:100])
    fig, ax = plt.subplots(3,3, figsize = (10,10))
    axes = ax.flatten()
    for i in range(9):
        _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
        axes[i].imshow(shuff[i], cmap="Greys")
    plt.show()


#commented out because it doesn't work properly    
#def plot_letter_distribution(X,y):
#    count = np.zeros(26, dtype='int')
#    for i in y:
#        count[i] += 1
#    alphabets = []
#    for i in AZ_MAP:
#        alphabets.append(i)
#    fig,ax = plt.subplots(1,1, figsize=(10,10))
#    ax.barh(alphabets,count)
#    plt.xlabel("Number of elements ")
#    plt.ylabel("Characters")
#    plt.grid()
#    plt.show()
#    
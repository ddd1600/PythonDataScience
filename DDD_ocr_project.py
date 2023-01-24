#coursera add-ons
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid, relu, linear

#packages for importing my own hand drawn characters
import PIL.ImageOps
from PIL import Image
import cv2

#trying to improve the model using this post: https://data-flair.training/blogs/handwritten-character-recognition-neural-network/
from keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

#import logging #honest I'm not sure what this does
#logging.getLogger('tensorflow').setLevel(logging.ERROR) #or this..
#tf.autograph.set_verbosity(0) #or this...

#pyimagesearch add-ons
#from tensorflow.keras.datasets import mnist #import MNIST (0-9) dataset

#additional
from tempfile import TemporaryFile

ALPHABET_MAP = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def test_sample(x, model):
    n = x.shape[0] #expects x.shape = (784,)
    x = x.reshape(1,n) #the flattener! actually this just reaffirms the identity of x as a flat row of values
    prediction = model.predict(x)
    prediction_p = tf.nn.softmax(prediction)
    prediction_p = np.array(prediction_p)[0] #the predictions are nested inside of an empty array
    print(f" predictions:\n{prediction_p}")
    top_prediction_p_indices  = np.argsort(-prediction_p)[:3] #return the indices of the highest 3 values for probability P
    x = x.reshape((28,28))
    for i in range(3):
        print("i=", i)
        prediction_p_i = top_prediction_p_indices[i]
        print("prediction_p_i.type=", type(prediction_p_i), "   prediction_p_i=", prediction_p_i)   
        predicted_letter = ALPHABET_MAP[prediction_p_i]
        pct_possibility =  prediction_p[prediction_p_i] * 100.0
        print("the algo says there is a ", pct_possibility, "%% chance the letter shown is a ", predicted_letter, "\n\n")
    plt.imshow(x)#, interpolation='nearest')        
    plt.show()
    
def letter_d(fn="D.jpeg"):
    img_array = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img_array)
    img_pil = PIL.ImageOps.invert(img_pil)# inverts color scheme
    img28 = np.array(img_pil.resize((28,28), Image.ANTIALIAS))
    img_array = (img28.flatten())
    return img_array

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()
    
def save_model(model, fn="ocr_model.keras"):
    model.save(fn)
    
def load_existing_model(fn="ocr_model.keras"):
    return tf.keras.models.load_model(fn)

def go():
    X,y = get_data()
    model = load_existing_model()
    test_sample(X[15000], model)
    return X,y,model
    

def go_original(num_epochs=10):
    print("fetching data...")
    X,y = get_data()
    print("building model...")
    model = get_model(X,y)
    print("compiling model...")
    compile_model(model)
    print('num_epochs=', num_epochs)        
    history = model.fit(X,y,epochs=num_epochs)
    save_model(model)
    return (model, history)
    

def get_data():
    X = np.load("X_ocr_set.npy")
    y = np.load("y_ocr_set.npy")
    return(X,y)
    
def get_new_model(): #DOESN'T WORK CURRENTLY. MOVING TO ITS OWN FILE
    #per https://data-flair.training/blogs/handwritten-character-recognition-neural-network/
    m = Sequential(
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
    m.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    ) 
    return m
    
def get_model(X,y):
    m,n = X.shape
    model = Sequential(
        [
            tf.keras.Input(shape=(n,)),
            Dense(60, activation='relu'),
            Dense(40, activation='relu'),
            Dense(26, activation='linear')
        ], name="hold_on_to_your_butts"
    )
    return model
    
def compile_model(model):
    model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        )






def initial_load_data(datasetPath="./handwritten_characters_db/A-Z/A_Z Handwritten_Data.csv", reshape=False): #code from pyimagesearch
    data = []
    labels = []
    
    for row in open(datasetPath):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype='uint8') #I'm going to write some code below that I think does this less succinctly:
        #image = np.array([])
        #for x in row[1:]:
        #    image.append(int(x), dtype='uint8') #not sure if this would do the same as the one liner. Might be worth checking out
        
        if reshape == True:
            image = image.reshape((28,28)) #images are represented as flat arrays of 784 pixel values. Reshaping into pictures again here.
        data.append(image)
        labels.append(label)
    X = np.array(data, dtype='float32')
    y = np.array(labels, dtype='int32')
    print("saving data to npy format...")
    np.save("X_ocr_set.npy", X)
    np.save("y_ocr_set.npy", y)
    return (X, y)
    
def visualize_data(X,y):
    m,n = X.shape
    fig, axes = plt.subplots(8,8,figsize=(5,5))
    fig.tight_layout(pad=0.13, rect=[0,0.03,1,0.91])
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        X_random_reshaped = X[random_index].reshape((28,28)).T
        ax.imshow(X_random_reshaped)#, cmp='gray')
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
   
            

''' image classification using tensorflow and convolutional neural networks (CNN)

convolutional neural networks are a type of neural networks that you use when you're dealing
 with image data or audio data or whenever you want to find patterns in data
 '''



import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

import os

# Check if CONDA_PREFIX environment variable is defined
if 'CONDA_PREFIX' in os.environ:
    conda_env_path = os.environ['CONDA_PREFIX']
    print(f"The script is running within the Conda environment at: {conda_env_path}")
else:
    print("The script is not running within a Conda environment.")



# arrays of pixels
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
    plt.subplot(4,4,i+1) #we have a four times four grid and with each iteration we're choosing one of those one of these places in the grid to place the next image
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary) #cmap is color map
    plt.xlabel(class_names[training_labels[i][0]]) # getting the label of the image (number), then passing this number as the index for our class list

#plt.show()


# reduce training images in the neural network
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

'''model = models.Sequential() # define neural network, basic model
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape= (32, 32, 3))) # define input layer, convolutional layer, 32 neurons, (3,3) confolution matrix -> filter, activation function, input shape is 32 x 32 x 3 -> resolution 32x32 3 color channels
model.add(layers.MaxPooling2D((2,2))) #MaxPooling simplifies result and reduces it to essential information
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu')) # each of these convolutional layers will filter for features in an image

model.add(layers.Flatten()) # 1 dimensional 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10,activation='softmax')) #softmax scales all probabilites to add up to 1

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics= ['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))



Max pooling is a popular technique used in convolutional neural networks (CNNs) for down-sampling or reducing the spatial dimensions (width and height) of feature maps while retaining the most important information.


loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}") #
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')'''


model = models.load_model('image_classifier.keras')

img = cv.imread('C:/Users/mmmah/Image Classification/horse-561221_640.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #the model was trained on RBG but the images are BGR, basically were swapping red and blue

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255) # pass the image as a numpy array
index = np.argmax(prediction) # we get 10 activations of the 10 softmax neurons. we want the index of the "brightest" neuron. argmax returns the index of the largest activation 

print(f"Prediction is {class_names[index]}")







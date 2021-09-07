import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#import Fashion MNIST dataset from Tensorflow
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Lables/ Class names 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Explore the dataset
shape = train_images.shape
# print(shape)
ti_len = len(train_images)
# print(ti_len)


# Preprocess the dataset
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

#Scale these values to a range of 0 to 1 before feeding them to the neural network model

train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training set and display the class name below each image.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()


# Building the model. The basic building block of a neural network is the layer. 
# Layers extract representations from the data fed into them.  

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #  transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
    tf.keras.layers.Dense(128, activation='relu'), # These are densely connected, or fully connected, neural layers.  Has 128 nodes (or neurons).
    tf.keras.layers.Dense(10) # These are densely connected, or fully connected, neural layers. Output layer.
])

# Compile the Model.

'''
Before the model is ready for training, it needs a few more settings. 
These are added during the model's compile step:
 
-> Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
-> Optimizer —This is how the model is updated based on the data it sees and its loss function.
-> Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Training the model involves the following steps.

'''
1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
2. The model learns to associate images and labels.
3. You ask the model to make predictions about a test set—in this example, the test_images array.
4. Verify that the predictions match the labels from the test_labels array.

'''


# 1.Feed the model
model.fit(train_images, train_labels, epochs=10)

# 2. Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#3. Make PREDICTIONS

'''
With the model trained, you can use it to make predictions about some images. 
The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
'''

probability_model = tf.keras.Sequential([model, tf.keras.layers.softmax()])
predictions = probability_model.predict(test_images)
predictions[0]



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Methods
def plot_img(i, predictions_arr, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary) 
    predicted_label = np.argmax(predictions_arr)

    if predicted_label == true_label:
        color = 'Blue'
    else:
        color = 'Red'
    
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label], 
                                  100*np.max(predictions_arr),
                                  class_names[true_label]),
                                  color=color
                                  )

def plot_value_arr(i, predictions_arr, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_arr, color='Green')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_arr)
    thisplot[predicted_label].set_color('Red')
    thisplot[true_label].set_color('Blue')

#Main program
#print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_imgs.shape)
#print(len(train_labels))
#print(train_labels)
#print(test_imgs.shape)
#print(len(test_labels))
#print(test_labels)

#Plotting the img
'''
plt.figure()
plt.imshow(train_imgs[9])
plt.colorbar()
plt.grid(False)
plt.show()
'''

train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

'''
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])

model.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_imgs, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
#print('\nTest Accuracy: ', test_acc)
probability_model = tf.keras.Sequential([
                    model, tf.keras.layers.Softmax()
                    ])

predictions = probability_model.predict(test_imgs)

#print(predictions[0])
#print(np.argmax(predictions[0]))
#print(test_labels[1])

'''
#Plotting single image with prediction
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_img(i, predictions[i], test_labels, test_imgs)
plt.subplot(1, 2, 2)
plot_value_arr(i, predictions[i], test_labels)
plt.show()
'''


#Plotting several images with predictions
num_rows = 5
num_cols = 3
num_imgs = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_imgs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_img(i, predictions[i], test_labels, test_imgs)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_arr(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()


'''
#Use the trained model to predict a single image.
img = test_imgs[1]
#print(img.shape)
img = np.expand_dims(img, 0)
#print(img.shape)
predictions_single = probability_model.predict(img)
#print(predictions_single)
plot_value_arr(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=90)
#plt.show()
print(np.argmax(predictions_single[0]))
'''
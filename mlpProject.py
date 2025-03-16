import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


train_images = train_images.reshape((train_images.shape[0], 784))
test_images = test_images.reshape((test_images.shape[0], 784))

# 1 hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model = models.Sequential() #MLP

model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,))) # input
model.add(layers.Dense(128, activation='relu'))#hidden
model.add(layers.Dense(10, activation='softmax'))#output


model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}")

predictions = model.predict(test_images)
index = 0  
predicted_label = predictions[index].argmax()
image_reshaped = test_images[index].reshape(28, 28)
plt.imshow(image_reshaped, cmap='gray')
plt.title(f"Predicted Label: {predicted_label}")
plt.axis('off')  
plt.show()
print(f"Predicted label for the first image: {predicted_label}")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model

num_classes = len(list(names_dict)) # should be 19

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.conv1 = Conv2D(filters=32, kernel_size=(7,7), strides=(1, 1), padding='valid', activation=None)
    self.bn1 = BatchNormalization()
    self.relu1 = Activation('relu')

    # we can define max pooling layer once, but use multiple times, because it doesn't have learnable parameters
    self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')

    self.conv2 = Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='valid', activation=None)
    self.bn2 = BatchNormalization()
    self.relu2 = Activation('relu')

    # self.conv3 = ...
    # self.bn3 = ...
    # self.relu3 = ...
    self.conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='valid', activation=None)
    self.bn3 = BatchNormalization()
    self.relu3 = Activation('relu')

    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.dropout = Dropout(0.05)
    self.d2 = Dense(num_classes)

  def nn(self, x, training=False):
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.relu1(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.relu2(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x = self.relu3(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout(x)
    return x

  def call(self, x):
    x = self.nn(x)
    x = self.d2(x)
    return x

  def features(self, x):
    x = self.nn(x)
    return x

# Create an instance of the model
model = MyModel()

# loss_object = ...
# Choices:
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#Exp: if last dense layer of model doesnt have sigmoid activation then from_logits=True is needed.
#     loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#     loss_object = tf.keras.losses.BinaryCrossentropy((from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



EPOCHS = 70
train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  train_loss_list.append(train_loss.result())
  test_loss_list.append(test_loss.result())
  train_acc_list.append(train_accuracy.result())
  test_acc_list.append(test_accuracy.result())

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
  
  
plt.plot(train_acc_list, label='train accuracy')
plt.plot(test_acc_list, label='test accuracy')
plt.xlabel('training epoch')
plt.legend()


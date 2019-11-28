# !/usr/bin/env python
# coding:utf-8
# Author: Huangyu

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
print(x_train.shape[0])
x_train, x_test = x_train / 255.0, x_test / 255.0

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

x_train=tf.convert_to_tensor(x_train)
x_test=tf.convert_to_tensor(x_test)
y_train=tf.squeeze(y_train, axis=1)
y_test=tf.squeeze(y_test, axis=1)

image_gen_train = ImageDataGenerator(
    rescale=1,  # 归至0～1
    rotation_range=0,  # 随机0度旋转
    width_shift_range=0.1,  # 宽度偏移
    height_shift_range=0.1,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=1  # 将图像随机缩放到100％
)
image_gen_train.fit(x_train)


class VGGNet(Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.05)

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.05)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.05)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.05)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.05)

        self.flatten = Flatten()
        self.f1 = Dense(4096, activation='relu')
        self.d6 = Dropout(0.05)
        self.f2 = Dense(4096, activation='relu')
        self.d7 = Dropout(0.05)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)
        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)
        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y

model = VGGNet()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/cifar10.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 save_best_only=True,
                                                 verbose=2)
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=128), epochs=200,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

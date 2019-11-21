# !/usr/bin/env python
# coding:utf-8
# Author: Huangyu
# coding:utf-8
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
import matplotlib.pyplot as plt
import os

input_word = "abcdefghigklmnopqrstuvwxyz"
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
           'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n':13, 'o':14 ,
'p': 15, 'q': 16, 'r': 17, 's': 18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25}  # 单词映射到数值id的词典
training_set_scales = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
x_train=[]
y_train=[]
for i in range(4,25):
    x_train.append(training_set_scales[i-4:i])
    y_train.append(training_set_scales[i])
print(x_train)
print(y_train)
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)

model = tf.keras.Sequential([
    Embedding(26, 4, input_length=4),
    SimpleRNN(3, return_sequences=True, unroll=True),
    SimpleRNN(4, unroll=True),
    Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/abcde.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 # save_best_only=True,
                                                 verbose=1)

history = model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback], verbose=1)

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
loss = history.history['loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

############### predict #############

preNum = int(input("input the number of test alphabet:"))
for i in range(preNum):
    alphabet1 = input("input test alphabet:")
    alphabet = [[w_to_id[i] for i in alphabet1]]
    print(alphabet)
    result = model.predict(alphabet)

    pred = tf.argmax(result, axis=1)
    pred = int(pred)

    tf.print(alphabet1 + '->' + input_word[pred])

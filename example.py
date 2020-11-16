import tensorflow as tf
import numpy as np
from NN import Dense, Model
import matplotlib.pyplot as plt

##########################################
#get data
def onehot_encoding(x):
    encoded = np.zeros((x.size, x.max()+1))
    encoded[np.arange(x.size),x] = 1
    return encoded

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
num_pixels = train_x.shape[1] * train_x.shape[2]
train_x = train_x.reshape(train_x.shape[0],num_pixels).astype('float32')/255.
test_x = test_x.reshape(test_x.shape[0],num_pixels).astype('float32')/255.
train_y = onehot_encoding(train_y)
test_y = onehot_encoding(test_y)

##########################################
#construct the model
MLP = Model(0.1)
MLP.add(Dense(784,256,activation='relu'))
MLP.add(Dense(256,64,activation='relu'))
MLP.add(Dense(64,10,activation='None'))

train_label = train_y.argmax(axis=1)
test_label =test_y.argmax(axis=1)
train_acc = []
val_acc = []

##########################################
#fit the model
for epoch in range(50):
    print(epoch, MLP.fit(train_x,train_y, 64))
    train_acc.append((MLP.predict(train_x) == train_label).mean())
    val_acc.append((MLP.predict(test_x) == test_label).mean())

##########################################
#visualize
plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='val accuracy')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
import pickle
import tensorflow as tf


jittered_file = './traffic-signs-data/jittered.p'

with open(jittered_file, mode='rb') as f:
    jit = pickle.load(f)
X_train, y_train = jit['features'], jit['labels']
assert(len(X_train) == len(y_train))

print("Number of training examples after jittering =", len(X_train))

X_r_channel = X_train[:, :, :, 0:1]
X_g_channel = X_train[:, :, :, 1:2]
X_b_channel = X_train[:, :, :, 2:3]

X_train = ( X_r_channel + X_g_channel + X_b_channel) // 3

print("X_train (red) = ", X_r_channel[0])
print("X_grey = ", X_train[0])
print("X_grey shape = ", X_train.shape)
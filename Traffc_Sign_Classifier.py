import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


### Load data from Pickle files ###

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

# Distributions seem skewed with large outliers, so we should use jittering to
# "fill in" some of the gaps.
# See jitter_training_data.py for the functions and script - it takes a long time
# to manipulate 20,000+ images so I pickled the jittered training data after
# generating it.
jittered_file = './traffic-signs-data/jittered.p'

with open(jittered_file, mode='rb') as f:
    jit = pickle.load(f)
X_train, y_train = jit['features'], jit['labels']
assert(len(X_train) == len(y_train))

print("Number of training examples after jittering =", len(X_train))

# jittered_labels, jittered_counts = countOccurrences(y_train)
# figure5 = plt.figure(figsize=(16, 4))
# figure5.suptitle("Distribution of Classes in Jittered Training Dataset", fontsize=16)
# plt.bar(jittered_labels, jittered_counts)
# plt.show()

### Basic Summary of Dataset ###

n_train = len(X_train)              # Number of training examples
n_validation = len(X_valid)         # Number of validation examples
n_test = len(X_test)                # Number of testing examples.
image_shape = X_train[0].shape      # The shape of each traffic sign image
n_classes = len(set(y_train))       # Number of unique classes/labels there are in the dataset.

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration & visualization ###

# Obviously, we can't visualize 30,000+ images easily. Just shuffle then plot a random sample
# X_train, y_train = shuffle(X_train, y_train)
#
# # Take the first 100 images to plot a sampler
# X_disp = X_train[:100]
# y_disp = y_train[:100]
#
# figure1 = plt.figure(figsize=(16, 4))
# figure1.suptitle("100 samples from 'German Traffic Signs' Database", fontsize=16)
# for i in range(len(X_disp)):
#     figure1.add_subplot(5, 20, i + 1)
#     plt.imshow(X_disp[i])
#     plt.axis('off')
# plt.show()


# Calculate and plot distributions (occurrences) for training, validation and test sets
def countOccurrences(labels):
    uniques = list(set(labels))             # Get unique labels contained in input
    counts = []                             # Output container

    for lbl in uniques:
        # Add 1 to count for each occurence of a unique label
        counts.append(sum(1 for y in labels if y == lbl))

    return uniques, counts


# # Plot distributions for training, validation and test sets
# train_labels, train_counts = countOccurrences(y_train)
# figure2 = plt.figure(figsize=(16, 4))
# figure2.suptitle("Distribution of Classes in Training Dataset", fontsize=16)
# plt.bar(train_labels, train_counts)
# plt.show()
#
# valid_labels, valid_counts = countOccurrences(y_valid)
# figure3 = plt.figure(figsize=(16, 4))
# figure3.suptitle("Distribution of Classes in Validation Dataset", fontsize=16)
# plt.bar(valid_labels, valid_counts)
# plt.show()
#
# test_labels, test_counts = countOccurrences(y_test)
# figure4 = plt.figure(figsize=(16, 4))
# figure4.suptitle("Distribution of Classes in Test Dataset", fontsize=16)
# plt.bar(test_labels, test_counts)
# plt.show()


# # Distributions seem skewed with large outliers, so we should use jittering to
# # "fill in" some of the gaps.
# # See jitter_training_data.py for the functions and script - it takes a long time
# # to manipulate 20,000+ images so I pickled the jittered training data after
# # generating it.
# jittered_file = './traffic-signs-data/jittered.p'
#
# with open(jittered_file, mode='rb') as f:
#     jit = pickle.load(f)
# X_train, y_train = jit['features'], jit['labels']
# assert(len(X_train) == len(y_train))
#
# print("Number of training examples after jittering =", len(X_train))

# jittered_labels, jittered_counts = countOccurrences(y_train)
# figure5 = plt.figure(figsize=(16, 4))
# figure5.suptitle("Distribution of Classes in Jittered Training Dataset", fontsize=16)
# plt.bar(jittered_labels, jittered_counts)
# plt.show()


### Data Preprocessing for Neural Network ###

# # Greyscale images
# X_train = (X_train[:,:,:,0:1] + X_train[:,:,:,1:2] + X_train[:,:,:,2:3]) // 3
# X_valid = (X_valid[:,:,:,0:1] + X_valid[:,:,:,1:2] + X_valid[:,:,:,2:3]) // 3
# X_test = (X_test[:,:,:,0:1] + X_test[:,:,:,1:2] + X_test[:,:,:,2:3]) // 3

# # Update image shape
# image_shape = X_train[0].shape
# print("Updated image shape after greyscale:", image_shape)

# Normalization
X_train = X_train/128 - 1
X_valid = X_valid/128 - 1
X_test = X_test/128 - 1




### Model Architecture ###
# Based off LeNet architecture, with modified layer widths

# TRAINING PARAMETERS
EPOCHS = 12
BATCH_SIZE = 100
LEARN_RATE = 0.0006
# LeNet PARAMETERS
MU = 0                  # Arguments for tf.truncated_normal, mean of truncated distribution
SIGMA = 0.05            # Arguments for tf.truncated_normal, stddev of truncated distribution
IMG_CHANNELS = image_shape[2]  # RGB images so C == 3
CONV1_F_SIZE = 5        # Filter width and height for 1st convolution layer
CONV1_N_OUT = 20        # No. of filters (output depth) for 1st convolution layer
CONV1_STRIDE = 1        # Stride size for 1st convolution layer
POOL1_SIZE = 2          # Resolution for 1st Maxpool layer
CONV2_F_SIZE = 5        # Filter width and height for 2nd convolution layer
CONV2_N_OUT = 44        # No. of filters (output depth) for 2nd convolution layer
CONV2_STRIDE = 1        # Stride size for 2nd convolution layer
POOL2_SIZE = 2          # Resolution for 2nd Maxpool layer
FC1_N_OUT = 560         # Output width for 1st fully connected layer
FC2_N_OUT = 300         # Output width for 2nd fully connected layer

# calculated parameters
# unsure how maxpool behaves if it sees "leftovers". For now, since pooling ksize == stride == 2,
# just make sure the convolution outputs have even-numbered width and height.
POOL1_WD = (32 - (CONV1_F_SIZE-1)) // CONV1_STRIDE // POOL1_SIZE
POOL2_WD = (POOL1_WD - (CONV2_F_SIZE-1)) // CONV2_STRIDE // POOL2_SIZE
POOL1B_WD = POOL1_WD // POOL1_SIZE

# Modified LeNet CNN. Accepts a 32x32xC image as input
# Parameters are controlled above
def LeNet(x, mu=MU, sigma=SIGMA, c=IMG_CHANNELS,
          conv1size=CONV1_F_SIZE, conv1out=CONV1_N_OUT, conv1stride=CONV1_STRIDE, pool1size=POOL1_SIZE,
          conv2size=CONV2_F_SIZE, conv2out=CONV2_N_OUT, conv2stride=CONV2_STRIDE, pool2size=POOL2_SIZE,
          pool1_wd=POOL1B_WD, pool2_wd=POOL2_WD,
          fc1out=FC1_N_OUT, fc2out=FC2_N_OUT, output_n=n_classes):

    # Set up dimensions of weights and biases first
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([conv1size, conv1size, c, conv1out], mean=mu, stddev=sigma)),
        'wc2': tf.Variable(tf.truncated_normal([conv2size, conv2size, conv1out, conv2out], mean=mu, stddev=sigma)),
        'wd1': tf.Variable(tf.truncated_normal([pool1_wd*pool1_wd*conv1out + pool2_wd*pool2_wd*conv2out, fc1out],
                                               mean=mu, stddev=sigma)),
        'wd2': tf.Variable(tf.truncated_normal([fc1out, fc2out], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([fc2out, output_n], mean=mu, stddev=sigma))}

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([conv1out], mean=mu, stddev=sigma)),
        'bc2': tf.Variable(tf.truncated_normal([conv2out], mean=mu, stddev=sigma)),
        'bd1': tf.Variable(tf.truncated_normal([fc1out], mean=mu, stddev=sigma)),
        'bd2': tf.Variable(tf.truncated_normal([fc2out], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([output_n], mean=mu, stddev=sigma))}

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x48. ReLu activation
    c1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, conv1stride, conv1stride, 1], padding='VALID') + biases['bc1']
    conv1 = tf.nn.relu(c1)

    # Pooling. Input = 28x28x48. Output = 14x14x48.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1size, pool1size, 1], strides=[1, pool1size, pool1size, 1],
                           padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x44. ReLu activation
    c2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, conv2stride, conv2stride, 1], padding='VALID') + biases['bc2']
    conv2 = tf.nn.relu(c2)

    # Pooling. Input = 10x10x44. Output = 5x5x44.
    pool2 = tf.nn.max_pool(conv2, ksize=[1, pool2size, pool2size, 1], strides=[1, pool2size, pool2size, 1],
                           padding='VALID')
    pool2_flat = tf.layers.flatten(pool2)       # Flatten. Input = 5x5x44. Output = 1100.

    # Pool convolution layer 1 again to match the "amount of subsampling" that data takes when going through the 2nd
    # convolution layer. Input = 14x14x20. Output = 7x7x20.
    pool1b = tf.nn.max_pool(pool1, ksize=[1, pool1size, pool1size, 1], strides=[1, pool1size, pool1size, 1],
                            padding='VALID')
    pool1b_flat = tf.layers.flatten(pool1b)       # Flatten. Input = 7x7x20. Output = 980.

    # Layer 3: Fully Connected. Input = 980 + 1100 (takes the output from both Pool1 and Pool2 together).
    # Output = 500. ReLu activation
    conv_pools_flat = tf.concat((pool1b_flat, pool2_flat), axis=-1)
    fc1 = tf.add(tf.matmul(conv_pools_flat, weights['wd1']), biases['bd1'])
    conn1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 500. Output = 300. ReLu activation
    fc2 = tf.add(tf.matmul(conn1, weights['wd2']), biases['bd2'])
    conn2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 300. Output = 43. Linear Combination
    logits = tf.add(tf.matmul(conn2, weights['out']), biases['out'])

    return logits


def evaluate(X_data, y_data, batch_size=BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset : offset+batch_size], y_data[offset : offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Training Pipeline ###

# Parameter sweep (for automating testing)
# Element 1: Convolution 1 filter depth
# Element 2: Convolution 2 filter depth
# Element 3: Fully-connected layer 1 output width
# Element 4: Fully-connected layer 2 output width
params = [108, 108, 200, 100]

# for run_config in params:
# Input/Label placeholders
x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# Define loss optimizer
logits = LeNet(x, conv1out=params[0], conv2out=params[1], fc1out=params[2], fc2out=params[3])
print("Confirmation: Training run with settings:", params[0], params[1], params[2], params[3])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARN_RATE)
training_operation = optimizer.minimize(loss_operation)

# Evaluation after training
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

outputfname = './traffic-signs-data/' + \
              str(params[0]) + '-' + str(params[1]) + '-' + \
              str(params[2]) + '-' + str(params[3]) + '.txt'
with open(outputfname, 'w') as file:  # Use file to refer to the file object
    file.write(outputfname)
    file.write('\r\n')

# Helper to save parameters after training
saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    # print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_train[offset : offset+BATCH_SIZE], y_train[offset : offset+BATCH_SIZE]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        test_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

        # Write results to file one epoch at a time
        with open(outputfname, 'a') as file:
            file.write("EPOCH {} ...".format(i + 1))
            file.write('\r\n')
            file.write("Test Accuracy = {:.3f}".format(test_accuracy))
            file.write('\r\n')
            file.write("Validation Accuracy = {:.3f}".format(validation_accuracy))
            file.write('\r\n')

    saver.save(sess, './lenet')
    print("Model saved")


# Evaluation on Test data
# Only do this at the end once things have been trained!
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
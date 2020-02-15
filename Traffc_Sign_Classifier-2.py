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
EPOCHS = 10
BATCH_SIZE = 100
LEARN_RATE = 0.001
# LeNet PARAMETERS
MU = 0                          # Arguments for tf.truncated_normal, mean of truncated distribution
SIGMA = 0.05                    # Arguments for tf.truncated_normal, stddev of truncated distribution
IMG_CHANNELS = image_shape[2]   # RGB images so C == 3

CONV1_F_SIZE = 5                # Filter width and height for 1st convolution layer
CONV1_N_OUT = 48                # No. of filters (output depth) for 1st convolution layer
POOL1_SIZE = 2                  # Resolution for 1st Maxpool layer

CONV2A_F_SIZE = 1               # Filter width and height for 1x1 convolution layer
CONV2A_N_OUT = 16               # No. of filters (output depth) for 1x1 convolution layer
POOL2A_SIZE = 2                 # Resolution for Maxpool of 1x1 convolution layer

CONV2B_F_SIZE = 5              # Filter width and height for 5x5 convolution layer
CONV2B_N_OUT = 48               # No. of filters (output depth) for 5x5/3x3 convolution layer
POOL2B_SIZE = 2                 # Resolution for Maxpool of 5x5/3x3 convolution layer

FC1_N_OUT = 200                 # Output width for 1st fully connected layer
FC2_N_OUT = 100                 # Output width for 2nd fully connected layer
FC_KEEP_RATE = 0.5              # Keep rate for dropout

# calculated parameters
# unsure how maxpool behaves if it sees "leftovers". For now, since pooling ksize == stride == 2,
# just make sure the convolution outputs have even-numbered width and height.
POOL1_WD = (32 - (CONV1_F_SIZE-1)) // POOL1_SIZE                # == 14
POOL2A_WD = (POOL1_WD - (CONV2A_F_SIZE-1)) // POOL2A_SIZE       # == 7
POOL2B_WD = (POOL1_WD - (CONV2B_F_SIZE-1)) // POOL2B_SIZE       # == 5


# Modified LeNet CNN. Accepts a 32x32xC image as input
# Parameters are controlled above
def LeNet(x, keep_prob, mu=MU, sigma=SIGMA, c=IMG_CHANNELS,
          conv1size=CONV1_F_SIZE, conv1out=CONV1_N_OUT, pool1size=POOL1_SIZE,
          conv2asize=CONV2A_F_SIZE, conv2aout=CONV2A_N_OUT, pool2asize=POOL2A_SIZE,
          conv2bsize=CONV2B_F_SIZE, conv2bout=CONV2B_N_OUT, pool2bsize=POOL2B_SIZE,
          pool2a_wd=POOL2A_WD, pool2b_wd=POOL2B_WD,
          fc1out=FC1_N_OUT, fc2out=FC2_N_OUT, output_n=n_classes):

    # Set up dimensions of weights and biases first

    # == 7x7x24 + 5x5x48, input width for the 1st fully connected layer
    fc1_width = pool2a_wd * pool2a_wd * conv2aout + pool2b_wd * pool2b_wd * conv2bout

    weights = {
        'wc1': tf.Variable(tf.truncated_normal([conv1size, conv1size, c, conv1out], mean=mu, stddev=sigma)),
        'wc2a': tf.Variable(tf.truncated_normal([conv2asize, conv2asize, conv1out, conv2aout],
                                                mean=mu, stddev=sigma)),
        'wc2b': tf.Variable(tf.truncated_normal([conv2bsize, conv2bsize, conv2aout, conv2bout],
                                                 mean=mu, stddev=sigma)),
        'wd1': tf.Variable(tf.truncated_normal([fc1_width, fc1out], mean=mu, stddev=sigma)),
        'wd2': tf.Variable(tf.truncated_normal([fc1out, fc2out], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([fc2out, output_n], mean=mu, stddev=sigma))}

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([conv1out], mean=mu, stddev=sigma)),
        'bc2a': tf.Variable(tf.truncated_normal([conv2aout], mean=mu, stddev=sigma)),
        'bc2b': tf.Variable(tf.truncated_normal([conv2bout], mean=mu, stddev=sigma)),
        'bd1': tf.Variable(tf.truncated_normal([fc1out], mean=mu, stddev=sigma)),
        'bd2': tf.Variable(tf.truncated_normal([fc2out], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([output_n], mean=mu, stddev=sigma))}

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x48. ReLu activation
    c1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    conv1 = tf.nn.relu(c1)
    # Pooling. Input = 28x28x48. Output = 14x14x48.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1size, pool1size, 1], strides=[1, pool1size, pool1size, 1],
                           padding='VALID')

    # Layer 2: Inception model. Conv 1x1 -> poolx2, Conv 1x1 -> 5x5 -> poolx2, and Conv 1x1 -> 3x3 -> poolx2
    # will all be fed to the first fully connected layer.

    # Layer 2a: Convolutional 1x1 -> poolx2. Input = 14x14x48. Output = 14x14x24. ReLu activation
    c2a = tf.nn.conv2d(pool1, weights['wc2a'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc2a']
    conv2a = tf.nn.relu(c2a)
    # Pooling. Input = 14x14x24. Output = 7x7x24.
    pool2a = tf.nn.max_pool(conv2a, ksize=[1, pool2asize, pool2asize, 1], strides=[1, pool2asize, pool2asize, 1],
                            padding='VALID')

    # Layer 2b: Convolutional 1x1 -> 5x5 -> poolx2. Input = 14x14x24. Output = 10x10x48. ReLu activation
    c2b1 = tf.nn.conv2d(conv2a, weights['wc2b'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc2b']
    conv2b = tf.nn.relu(c2b1)
    # Pooling. Input = 10x10x48. Output = 5x5x48.
    pool2b = tf.nn.max_pool(conv2b, ksize=[1, pool2bsize, pool2bsize, 1], strides=[1, pool2bsize, pool2bsize, 1],
                             padding='VALID')

    # Layer 3: Fully Connected. Input = 7x7x24 + 5x5x48. Output = 200. ReLu activation
    # Dropout implemented here reduces validation error significantly
    pool2a_flat = tf.layers.flatten(pool2a)
    pool2b_flat = tf.layers.flatten(pool2b)
    conv_pools_flat = tf.concat((pool2a_flat, pool2b_flat), axis=-1)
    fc1 = tf.add(tf.matmul(conv_pools_flat, weights['wd1']), biases['bd1'])
    conn1 = tf.nn.relu(fc1)
    conn1 = tf.nn.dropout(conn1, keep_prob)

    # Layer 4: Fully Connected. Input = 200. Output = 100. ReLu activation
    # Dropout implemented here reduces validation error significantly
    fc2 = tf.add(tf.matmul(conn1, weights['wd2']), biases['bd2'])
    conn2 = tf.nn.relu(fc2)
    conn2 = tf.nn.dropout(conn2, keep_prob)

    # Layer 5: Fully Connected. Input = 100. Output = 43. Linear Combination
    logits = tf.add(tf.matmul(conn2, weights['out']), biases['out'])

    return logits


def evaluate(X_data, y_data, dropout=1.0, batch_size=BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset : offset+batch_size], y_data[offset : offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Training Pipeline ###

# Parameter sweep (for automating testing)
# Element 1: 5x5 1st convolution filter depth
# Element 2: 1x1 2nd convolution filter depth
# Element 2: 5x5 2nd convolution filter depth
# Element 3: Fully-connected layer 1 output width
# Element 4: Fully-connected layer 2 output width
params = [48, 28, 48, 200, 100]

# for run_config in params:
# Input/Label placeholders
x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)

# Define loss optimizer
logits = LeNet(x, keep_prob,
               conv1out=params[0], conv2aout=params[1], conv2bout=params[2], fc1out=params[3], fc2out=params[4])
print("Confirmation: Training run with settings:", params[0], params[1], params[2], params[3], params[4])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARN_RATE)
training_operation = optimizer.minimize(loss_operation)

# Evaluation after training
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

outputfname = './traffic-signs-data/' + \
              str(params[0]) + '-' + str(params[1]) + '-' + str(params[2]) + '-' + \
              str(params[3]) + '-' + str(params[4]) + '.txt'
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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: FC_KEEP_RATE})

        test_accuracy = evaluate(X_train, y_train, 1.0)
        validation_accuracy = evaluate(X_valid, y_valid, 1.0)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(test_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

        # Write results to file one epoch at a time
        with open(outputfname, 'a') as file:
            file.write("EPOCH {} ...".format(i + 1))
            file.write('\r\n')
            file.write("Training Accuracy = {:.3f}".format(test_accuracy))
            file.write('\r\n')
            file.write("Validation Accuracy = {:.3f}".format(validation_accuracy))
            file.write('\r\n')

    saver.save(sess, './lenet')
    print("Model saved")


# Evaluation on Test data
# Only do this at the end once things have been trained!
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test, 1.0)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    # Write results to file one epoch at a time
    with open(outputfname, 'a') as file:
        file.write('Model saved\r\n')
        file.write("Test Accuracy = {:.3f}".format(test_accuracy))
        file.write('\r\n')
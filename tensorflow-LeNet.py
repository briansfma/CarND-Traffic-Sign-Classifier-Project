import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# Download and bin data into training/validation/test sets
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# For MNIST data -> LeNet ONLy
# MNIST images are all 28x28px, whereas our LeNet implementation takes 32x32px as the input
# So we pad images with 0s to compensate, which also helps center the character in frame
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

# Visualize data
index = np.random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

# Shuffle training data
X_train, y_train = shuffle(X_train, y_train)


# PARAMETERS
EPOCHS = 10
BATCH_SIZE = 128
MU = 0              # Arguments for tf.truncated_normal, mean of truncated distribution
SIGMA = 0.1         # Arguments for tf.truncated_normal, stddev of truncated distribution
LEARN_RATE = 0.001


# LeNet CNN implementation. Accepts a 32x32xC image as input; MNIST images are greyscale so C is just 1 here.
def LeNet(x, mu=MU, sigma=SIGMA):
    # TODO: Figure out dimensions first before doing anything
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma)),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'wd1': tf.Variable(tf.truncated_normal([5 * 5 * 16, 120], mean=mu, stddev=sigma)),
        'wd2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([84, 10], mean=mu, stddev=sigma))}

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sigma)),
        'bc2': tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sigma)),
        'bd1': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma)),
        'bd2': tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.truncated_normal([10], mean=mu, stddev=sigma))}

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6. ReLu activation
    c1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    conv1 = tf.nn.relu(c1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16. ReLu activation
    c2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc2']
    conv2 = tf.nn.relu(c2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool2_flat = tf.layers.flatten(pool2)     # Flatten. Input = 5x5x16. Output = 400.

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120. ReLu activation
    fc1 = tf.add(tf.matmul(pool2_flat, weights['wd1']), biases['bd1'])
    conn1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84. ReLu activation
    fc2 = tf.add(tf.matmul(conn1, weights['wd2']), biases['bd2'])
    conn2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10. Linear Combination
    logits = tf.add(tf.matmul(conn2, weights['out']), biases['out'])

    return logits


# Start Training Pipeline
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARN_RATE)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Start Tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")


# # Evaluation on Test data
# # Only do this at the end once things have been trained!
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

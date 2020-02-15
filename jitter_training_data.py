import pickle
import numpy as np
import cv2
from random import sample


# Calculate and plot distributions (occurrences) for training, validation and test sets
def countOccurrences(labels):
    uniques = list(set(labels))             # Get unique labels contained in input
    counts = []                             # Output container

    for lbl in uniques:
        # Add 1 to count for each occurrence of a unique label
        counts.append(sum(1 for y in labels if y == lbl))

    return uniques, counts


# Generate randomly perturbed images and insert them into the set
def add_jitter_images(image, img_set, n_copies):
    jmethods = sample(range(6), n_copies)

    rows, cols = image.shape[:2]
    for j in jmethods:
        if j == 0:
            # shift image 2px down and right
            translation_matrix = np.float32([[1, 0, 2], [0, 1, 2]])
            fix_img = cv2.warpAffine(image, translation_matrix, (cols, rows))
        if j == 1:
            # shift image 2px up and left
            translation_matrix = np.float32([[1, 0, -2], [0, 1, -2]])
            fix_img = cv2.warpAffine(image, translation_matrix, (cols, rows))
        if j == 2:
            # rotate image by 15deg CCW
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
            fix_img = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        if j == 3:
            # rotate image by 15deg CW
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
            fix_img = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        if j == 4:
            # scale image by 0.9x (down to 28x28px)
            temp = cv2.resize(image, (28, 28))
            # pad image back to 32x32px
            fix_img = cv2.copyMakeBorder(temp, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        if j == 5:
            # scale image by 1.1x (up to 35x35px)
            temp = cv2.resize(image, (35, 35))
            # crop image back to 32x32px
            fix_img = temp[1:33, 1:33]

        img_set = np.concatenate((img_set, [fix_img]), axis=0)

    return img_set


### Load data from Pickle files ###

training_file = './traffic-signs-data/train.p'
jittered_file = './traffic-signs-data/jittered.p'

# Open and extract training data
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']
assert(len(X_train) == len(y_train))

# Get distributions for training set - will need this to calculate which labels' images we need
# to jitter (to generate more data)
train_labels, train_counts = countOccurrences(y_train)

print("y_train:", y_train)
print("train_labels:", train_labels)
print("train_counts:", train_counts)

# Loop through training set to identify which labels need more data
jitter_Xs = X_train
jitter_ys = y_train
for i in range(len(train_labels)):
    # Severe deficiency -- label occurrence is < 1/4 of the highest
    if train_counts[i] < max(train_counts)/4:
        print("Insuff (<1/4): ", train_labels[i])
        for x, y in zip(X_train, y_train):
            if y == train_labels[i]:
                # Add 5 jittered images per image corresponding to this label (effective x6 to data)
                jitter_Xs = add_jitter_images(x, jitter_Xs, 5)
        jitter_ys = np.append(jitter_ys, np.full((5*train_counts[i]), train_labels[i]))
    # Mild deficiency -- label occurrence is < 1/2 of the highest
    elif train_counts[i] < max(train_counts)/2:
        print("Insuff (<1/2): ", train_labels[i])
        for x, y in zip(X_train, y_train):
            if y == train_labels[i]:
                # Add 3 jittered images per image corresponding to this label (effective x4 to data)
                jitter_Xs = add_jitter_images(x, jitter_Xs, 3)
        jitter_ys = np.append(jitter_ys, np.full((3*train_counts[i]), train_labels[i]))
    # No big deficiency -- label occurrence is within 50% of the highest
    else:
        print("Within <1/2: ", train_labels[i])
        for x, y in zip(X_train, y_train):
            if y == train_labels[i]:
                # Add 1 jittered image per image corresponding to this label (effective x2 to data)
                jitter_Xs = add_jitter_images(x, jitter_Xs, 1)
        jitter_ys = np.append(jitter_ys, np.full((train_counts[i]), train_labels[i]))

print("Jitter_Xs' size: ", jitter_Xs.shape)
print("Jitter_ys' size: ", jitter_ys.shape)

# Verify distribution of jittered dataset - should be much less peaky than original
train_labels, train_counts = countOccurrences(jitter_ys)
print("train_counts:", train_counts)

# Make dictionary to pickle new training data
train_jittered = {
    'features': jitter_Xs,
    'labels':   jitter_ys
}

# Save pickle file
with open(jittered_file, mode='wb') as f:
    pickle.dump(train_jittered, f)


# Double-check: read and extract pickle file
with open(jittered_file, mode='rb') as f:
    jit = pickle.load(f)

X_jit, y_jit = jit['features'], jit['labels']

print("X_jit = ", X_jit)
print("X_jit size = ", X_jit.shape)
print("y_jit = ", y_jit)
print("y_jit size = ", y_jit.shape)

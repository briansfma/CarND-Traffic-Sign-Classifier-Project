# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-files/visualization.jpg "Visualization"
[image2]: ./writeup-files/grayscale.jpg "Grayscale"
[image3]: ./writeup-files/jittered.jpg "Jittered"
[image4]: ./writeup-files/jittered-dist.jpg "New distribution"
[image5]: ./writeup-files/online_signs.jpg "From Web"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is a link to my [project code](https://github.com/briansfma/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Python and pandas methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is len(X_train), or 34799
* The size of the validation set is len(X_valid), or 4410
* The size of test set is len(X_test), or 12630
* The shape of a traffic sign image is X_train[0].shape, or (32, 32, 3)
* The number of unique classes/labels in the data set is len(set(y_train)), or 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The biggest observation is that the distributions of labels aren't remotely close to even - some examples are almost 10x more common than others, which can't help for accuracy on the less-common signs.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried to convert the images to grayscale because Sermanet & LeCun reported greater accuracy doing so. I personally did not, so, although the code to grayscale the images is there, I commented it out as I could not achieve >90% accuracy using grayscale.

Here is an example of a few images from the Training set after grayscaling. Again, these were NOT used for final training.

![alt text][image2]

At the end of the day, I made two preprocessing steps:
1) Generate "jittered" images in the vein of Sermanet & LeCun
2) Normalized images before feeding them to the network

I decided to generate additional data because the jittering would increase the number of samples for relatively "rare" labels. Due to a shortage of computational power (my laptop is 10 years old at this time) I had to limit the number of jittered images, so rare labels received more (between 3 extra or 5 extra images) of this treatment, and common labels received less (only 1 extra).

To add more data to the the data set, I wrote some functions and a script in `jitter_training_data.py` to create and add jittered images to the Training set:

```
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
```
The code reflects six different ways of slightly altering an original image:
1) Shift image 2px down and right
2) Shift image 2px up and left
3) Rotate image 15deg CCW
4) Rotate image 15deg CW
5) Scale image by 0.9x
6) Scale image by 1.1x

Which technique used exactly was chosen at random, with no overlaps. 

Here is an example of an original image (left) and 5 of the 6 jittering techniques applied (one of the "rare" labels from the Training set).

![alt text][image3]

As it takes a while for it to run, I wrote it as a separate Python script and pickled the resulting data into `./traffic-signs-data/jittered.p` (I'm not sure this file will make it on Github though as it's larger than 100MB).

The distributions have been "filled in" somewhat by the jittering operation, and the new number of training examples = 101936.

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer     			|     Description       						| 
|:---------------------:|:---------------------------------------------:| 
| 0) Input 				| 32x32x3 RGB image   							| 
|						|												|
| 1) Convolution 5x5 	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU 					|												|
| Max pooling 			| 2x2 stride,  outputs 14x14x48 				|
|						|												|
| 2) Inception  		|												|
| 2a) Convolution 1x1	| 1x1 stride, valid padding, outputs 14x14x32 	|
| RELU 					|												|
| Max pooling 			| 2x2 stride,  outputs 7x7x32 					|
| 2b) Convolution 1x1 	| 1x1 stride, valid padding, outputs 14x14x32 	|
| RELU 					|												|
| Convolution 5x5 		| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU 					|												|
| Max pooling 			| 2x2 stride,  outputs 5x5x48 					|
|						|												|
| 3) Fully Connected 	| Outputs 200 wide 								|
| RELU 					|												|
| Dropout 				| Keep rate 50% 								|
|						|												|
| 4) Fully Connected 	| Outputs 100 wide 								|
| RELU 					|												|
| Dropout 				| Keep rate 50% 								|
|						|												|
| 5) Output 			| Outputs 43 wide 								|
|						|												|
|						|												|
 
This model ended up being more inspired by GoogLeNet than LeNet - I was not prepared hardware-wise (4GB RAM, ancient CPU, integrated graphics) so I needed to minimize the number of operations during training to even complete this lab. The 1x1 convolutions used in the inception layer turned out to be a godsend, making this model much faster than the "layer skipping LeNet" implementation Sermanet & LeCun described while still reaching 97.1% accuracy.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I did not do anything special for training - I used the same cross entropy-based loss operation as taught in class, the same AdamOptimizer as given in example code during labs. Batch size was set to 100, data was shuffled before batching, and learning rate was set to 0.001. I experimented with larger and smaller values for batch size and learning rate, but found no significant gains deviating from these round numbers.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.3%
* test set accuracy of 97.1%

I used an iterative approach to build my network - using a pre-made network is something I would do on the job, but this exercise is for learning how they work, so it makes more sense to get dirty and break things.

* What was the first architecture that was tried and why was it chosen?

The first architecture tried was LeNet, as it trains quickly and it reached 92% validation accuracy without too much effort.

* What were some problems with the initial architecture?

It seemed like, no matter what I changed parameter-wise, LeNet either overfits (>7% delta between training and validation accuracy) or just was not very accurate. I could never top 93% reliably with LeNet.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Since the network could pretty easily reach 99% accuracy on the training set, the overfitting delta between training and validation accuracy became a key metric to minimize for overall system performance. I experimented with dataset construction and parameter tuning before giving up on LeNet, then implemented Sermanet's traffic sign classifier architecture to try putting both 1st and 2nd-layer convolutions into the fully connected layers. The layer-skipping approach decreased the overfit delta to around 5-6%, and I did notice that performance improved as the number of parameters grew. But, it slowed down training by over 10x and I capped out around 95% validation accuracy.

While waiting for training experiments to complete, I skipped ahead in course materials and learned about GoogLeNet and its focus on small size and fast speed. The inception module was added in, originally with 3 subparts (1x1 -> pool, 1x1 -> 5x5 -> pool and 1x1 -> 3x3 -> pool just like the course materials describe) but I found no difference if I just removed the 3x3. So with a (1x1 -> pool, 1x1 -> 5x5 -> pool) inception layer I saw the overfit delta decrease to 4% at best.

I realized very late that the two fully connected layers were a potential source of overfitting. Both of these were changed to dropout layers, and this made the big difference I needed, with the overfit delta dropping to 1.6-1.8% at best. Still not perfect, but now good enough that if the test accuracy is 99.9%, the validation accuracy can be above 98%, and test accuracy can now be above 97%.

For those interested, the iterative progress (training accuracy, validation accuracy, etc across epochs) is logged in several text files within the extra folders in this Git (`./Runs with inception and dropout`, etc).

* Which parameters were tuned? How were they adjusted and why?

I experimented with convolution size and stride early on, but did not find any gains straying away from a 5x5 convolution with 1px stride. It seemed like most of the gains were to be found in filter depths and layer widths so I focused my parameter tuning there.

To attempt to get the best performance possible out of the training epochs, I implemented parameter saving on the model only if the validation accuracy of the current epoch was better than all epochs past. This doesn't "help" the network per se but avoids losing progress if the training was going well and suddenly started performing worse.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Multiple convolution layers makes sense for this problem as we are trying to hunt for small features on portions of an image just a few pixels wide - having different sizes and stacks of convolution processes gives the network a chance to build the weights to identify the diverse features. Implementing dropout on the fully connected layers was empirically extremely helpful - perhaps because many of the signs share similar features, differing only by a pattern in the center, it would be useful to drop out nodes that activate on commonly shared features (a red ring, for instance) and focus training on the nodes that activate with distinguishing features.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The second and fifth images might be mis-classified if the network latches on to the portion of the other sign also in the image. Overall, the lighting and clarify of these images is better than the original Training dataset though, so I did not really expect this to give the network too much trouble.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image      			|     Prediction        						| 
|:---------------------:|:---------------------------------------------:| 
| Do Not Enter  		| (17) Do Not Enter  							| 
| Double Curve  		| (21) Double Curve  							|
| Road Narrows on Right | (24) Road Narrows on Right 					|
| Priority Road 		| (12) Priority Road 			 				|
| Road Work 			| (25) Road Work 								|
| 30 km/h 				| (01) 30 km/h 									|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. As expected, the model is quite sure (99%) of its guesses, and it is correct.

For the first image...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00 		| (17) Do Not Enter 							| 
| 5.8687019e-24 		| (14) Stop 									|
| 6.3757657e-31 		| (10) No passing (for vehicles over 3.5T) 		|
| 8.0511211e-38 		| (20) Dangerous curve to the right 			|
| 0.0000000e+00 		| (0) Speed Limit 20 km/h 						|

For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00 		| (21) Double Curve 							| 
| 2.8207174e-11 		| (11) Right-of-way at the next intersection 	|
| 1.9636484e-16 		| (24) Road narrows on the right 				|
| 3.2241703e-18 		| (31) Wild animals crossing 					|
| 4.0319966e-20 		| (19) Dangerous curve to the left 				|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9999201e-01 		| (24) Road narrows on the right 				|
| 7.9505771e-06 		| (27) Pedestrians 								|
| 2.4371257e-11 		| (18) General caution 							|
| 1.3620277e-12 		| (21) Double Curve 							| 
| 4.8079151e-13 		| (19) Dangerous curve to the left 				|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00 		| (12) Priority road 							| 
| 8.7379896e-20 		| (10) No passing (for vehicles over 3.5T) 		|
| 6.5818561e-20 		| (42) End of no passing (vehicles over 3.5T) 	|
| 2.7271657e-20 		| (13) Yield 									|
| 1.2722329e-20 		| (17) Do Not Enter 							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000000e+00 		| (25) Road Work 								| 
| 1.0086585e-15 		| (24) Road narrows on the right 				|
| 4.1117444e-18 		| (29) Bicycles crossing 						|
| 1.7903638e-18 		| (30) Beware of ice/snow 						|
| 1.5712872e-18 		| (18) General caution 							|

For the sixth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9919802e-01 		| (1) Speed Limit 30 km/h 						|
| 8.0016337e-04 		| (2) Speed Limit 50 km/h 						|
| 1.7857768e-06 		| (13) Yield 									|
| 5.8697630e-11 		| (15) No vehicles 								|
| 1.5025597e-11 		| (32) End of all speed and passing limits 		|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



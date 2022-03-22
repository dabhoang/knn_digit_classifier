import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from knn import *

#Loading the data
M = loadmat('MNIST_digit_data.mat')
images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']

#just to make all random sequences on all computers the same.
np.random.seed(1)

#randomly permute data points
inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]

#if you want to use only the first 1000 data points.
#images_train = images_train[0:1000,:]
#labels_train = labels_train[0:1000,:]


#plots

new_training_images = images_train[0:1000]
new_training_labels = labels_train[0:1000]

new_validation_images = images_train[1000:2000]
new_validation_labels = labels_train[1000:2000]

k_values = [1,2,3,5,10]
results_validation = []
for k_value in k_values:
    accuracy = kNN(new_training_images, new_training_labels, new_validation_images, new_validation_labels, k_value)
    results_validation.append(accuracy[1])
plt.plot(k_values, results_validation)

plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.title('Accuracy by training size for different k')
plt.savefig('knn_validation.png')

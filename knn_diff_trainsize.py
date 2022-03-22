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
#4
training_sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]


k_values = [1,2,3,5,10]

for k_value in k_values:
    results_training_sizes = []
    for i in range(10):
        current_training_images = images_train[0:training_sizes[i],:]
        current_training_labels = labels_train[0:training_sizes[i],:]
        print('k = ' + str(k_value))
        print('training size = ' + str(len(current_training_images)))
        accuracy = kNN(current_training_images, current_training_labels, images_test, labels_test, k_value)
        results_training_sizes.append(accuracy[1])
    plt.plot(training_sizes, results_training_sizes, label='k = ' + str(k_value))

plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.title('Accuracy by training size for different k')
plt.legend()
plt.savefig('knn_4.png')


    

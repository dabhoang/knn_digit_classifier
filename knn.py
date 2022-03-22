from email.mime import image
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def kNN(images_train, labels_train, images_test, labels_test, k):
    #for test_image in images_test:
    #
    correct_predictions = [0,0,0,0,0,0,0,0,0,0]
    total_correct = 0
    predictions = []
    for i in range(len(images_test)):
    #for test_image in images_test:
        test_image = images_test[i]
        test_image = np.array(test_image)
        #compute euclidean distance from current test image to every training image
        distances = []

        for training_image in images_train:
            training_image = np.array(training_image)
            eucl_dist = np.linalg.norm(training_image - test_image)
            distances.append(eucl_dist)
            
        #print("Eucl distances between current testing image & first 10 training images: " + str(distances[0:10]))
        

        #get indices of k nearest neighbors to current test image
        distances = np.array(distances)
        distances_indices = np.argpartition(distances, k)

        #of k nearest neigbors,  count how many are classified into each label
        tallies = [0,0,0,0,0,0,0,0,0,0]

        for x in range(k):
            label = labels_train[distances_indices[x]][0]
            tallies[label]+=1
        tallies = np.array(tallies)
        #print(tallies)

        #predicted label
        predicted_label = np.argmax(tallies)

        #print("predicted label: " + str(predicted_label))
        #print("correct label: " + str(labels_test[i][0]))
        
        if predicted_label == labels_test[i][0]:
            correct_predictions[predicted_label]+=1
            total_correct+=1

    #accuracy metrics
    

    testing_count = [0,0,0,0,0,0,0,0,0,0]
    for label in labels_test:
        testing_count[label[0]]+=1

    acc = []
    for x in range(10):
        acc.append(correct_predictions[x]/testing_count[x])

    print("Accuracy by label:" + str(acc))
    
    acc_av = total_correct/len(labels_test)
    print("Overall accuracy: " + str(acc_av))
    
    return [acc, acc_av]
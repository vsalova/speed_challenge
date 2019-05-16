import numpy as np
import cv2
import random
import os
from optical_flow import optical_flow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

images = '../data/images'

print("\nConverting groundTruth labels to numpy array...")

data = []
with open('../data/train.txt', 'r') as f:
    for line in f:
        data.append(float(line.split()[0]))

# convert to numpy array
data = np.asarray(data)

# extract speed
y = data[1:4715]

X = []

#change this to be dynamic
for i in range(2):
    # print("\tProcessing img {}".format(i))
    # print("\tProcessing img {}".format(i+1))
    filepath1 = os.path.join(images ,"frame%d.jpg" % i)
    filepath2 = os.path.join(images ,"frame%d.jpg" % i)
    img1 = cv2.imread(filepath1, 0)
    img2 = cv2.imread(filepath2, 0)

    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

    plt.imshow(img3),plt.show()

















#
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
# xTrain = np.array(xTrain)
# xTest = np.array(xTest)
# yTrain = np.array(yTrain)
# yTest = np.array(yTest)
#
# xTrain = xTrain.reshape(-1, 1)
# xTest = xTest.reshape(-1, 1)
# yTrain = yTrain.reshape(-1, 1)
# yTest = yTest.reshape(-1, 1)
#
# print(xTrain.shape)
# print(xTest.shape)
# print(yTrain.shape)
# print(yTest.shape)
#
# reg = LinearRegression().fit(xTrain, yTrain)
#
# predicted = reg.predict(xTest)
#
# score = reg.score(predicted, yTest)
# comparison = yTest - predicted
#
# print(score)
#
# np.savetxt('test.txt', comparison)
# # plt.plot(predicted)
# # plt.plot(yTest)
# #
# # plt.savefig('test.png')
#
# print("\nDone!")

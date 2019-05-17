import numpy as np
import cv2
import random
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

train_images = '../data/train_images'
test_images = '../data/test_images'

print("\n TRAINING:")

print("\nConverting groundTruth labels to numpy array...")

data = []
with open('../data/train.txt', 'r') as f:
    for line in f:
        data.append(float(line.split()[0]))

# convert to numpy array
data = np.asarray(data)

# extract speed
yTrain = data[:]

xTrain = []


print("\nProcessing images...")

#change this to be dynamic
for i in range(20399):
# for i in range(7,9):
    # print("\tProcessing img {}".format(i))
    # print("\tProcessing img {}".format(i+1))
    filepath1 = os.path.join(train_images ,"frame%d.jpg" % i)
    filepath2 = os.path.join(train_images ,"frame%d.jpg" % (i+1))
    img1 = cv2.imread(filepath1, 0)
    img1 = img1[250:375, :]
    img2 = cv2.imread(filepath2, 0)
    img2 = img2[250:375, :]

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    try:
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Initialize lists
        frame1 = []
        frame2 = []
        distance = []
        # For each match...
        for match in matches:
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            # Append to each list
            frame1.append((x1, y1))
            frame2.append((x2, y2))

            for i in range(len(frame1)):
                # dist = math.sqrt((frame1[i][0]-frame2[i][0])**2+(frame1[i][1]-frame2[i][1])**2)
                dist = abs(frame1[i][1]-frame2[i][1])
                distance.append(dist)

    except cv2.error:
        pass

    if math.isnan(np.mean(distance)):
        xTrain.append(xTrain[-1])
    # elif np.mean(distance) == 0:
        # xTrain.append(xTrain[-1])
    else:
        xTrain.append(np.mean(distance))

    if len(xTrain) == 1:
        xTrain.append(np.mean(distance))

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10], None, flags=2)
    # plt.imshow(img3),plt.show()


print("\nFinished collecting data...")

print("\n\n\n")

print("\nPrepare and run linear regression...")

# get the pixel velocity
xTrain[:] = [dist / 0.05 for dist in xTrain]
#
# xTrain = np.array(X[:18360])
# xTest = np.array(X[18360:])
# yTrain = np.array(y[:18360])
# yTest = np.array(y[18360:])

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

xTrain = xTrain.reshape(-1, 1)
# xTest = xTest.reshape(-1, 1)
yTrain = yTrain.reshape(-1, 1)
# yTest = yTest.reshape(-1, 1)

reg = LinearRegression().fit(xTrain, yTrain)



print("\n TESTING:")

xTest = []

for i in range(10797):
# for i in range(7,9):
    # print("\tProcessing img {}".format(i))
    # print("\tProcessing img {}".format(i+1))
    filepath1 = os.path.join(test_images ,"frame%d.jpg" % i)
    filepath2 = os.path.join(test_images ,"frame%d.jpg" % (i+1))
    img1 = cv2.imread(filepath1, 0)
    img1 = img1[250:375, :]
    img2 = cv2.imread(filepath2, 0)
    img2 = img2[250:375, :]

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    try:
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Initialize lists
        frame1 = []
        frame2 = []
        distance = []
        # For each match...
        for match in matches:
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            # Append to each list
            frame1.append((x1, y1))
            frame2.append((x2, y2))

            for i in range(len(frame1)):
                # dist = math.sqrt((frame1[i][0]-frame2[i][0])**2+(frame1[i][1]-frame2[i][1])**2)
                dist = abs(frame1[i][1]-frame2[i][1])
                distance.append(dist)

    except cv2.error:
        pass

    if math.isnan(np.mean(distance)):
        xTest.append(xTest[-1])
    # elif np.mean(distance) == 0:
        # xTest.append(xTest[-1])
    else:
        xTest.append(np.mean(distance))

    if len(xTest) == 1:
        xTest.append(np.mean(distance))

xTest[:] = [dist / 0.05 for dist in xTest]

xTest = np.array(xTest)
xTest = xTest.reshape(-1, 1)
predicted = reg.predict(xTest)

# score = reg.score(predicted, yTest)
# comparison = yTest - predicted

# print(score)

# np.savetxt('../data/comparison.txt', comparison)
np.savetxt('../data/test.txt', predicted)
# plt.plot(predicted)
# plt.plot(yTest)
#
# plt.savefig('../data/test.png')

print("\nDone!")

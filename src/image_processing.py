import numpy as np
import cv2
import random
import os
from optical_flow import optical_flow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

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


print("\nProcessing images...")

#change this to be dynamic
for i in range(4714):
# for i in range(7,9):
    # print("\tProcessing img {}".format(i))
    # print("\tProcessing img {}".format(i+1))
    filepath1 = os.path.join(images ,"frame%d.jpg" % i)
    filepath2 = os.path.join(images ,"frame%d.jpg" % (i+1))
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
        print("mean: ", distance)

    X.append(np.mean(distance))

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10], None, flags=2)
    # plt.imshow(img3),plt.show()


print("\nFinished collecting data...")

print("\n\n\n")

print("\nPrepare and run linear regression...")

# get the pixel velocity
X[:] = [dist / 0.05 for dist in X]

xTrain = np.array(X[:3771])
xTest = np.array(X[3771:])
yTrain = np.array(y[:3771])
yTest = np.array(y[3771:])

xTrain = xTrain.reshape(-1, 1)
xTest = xTest.reshape(-1, 1)
yTrain = yTrain.reshape(-1, 1)
yTest = yTest.reshape(-1, 1)

print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

reg = LinearRegression().fit(xTrain, yTrain)

predicted = reg.predict(xTest)

score = reg.score(predicted, yTest)
comparison = yTest - predicted

print(score)

np.savetxt('test.txt', comparison)
plt.plot(predicted)
plt.plot(yTest)

plt.savefig('test.png')

print("\nDone!")

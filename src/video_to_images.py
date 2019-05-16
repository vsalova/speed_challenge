import os
import cv2

# Run this just once to convert the video into images

data = '../data/'
images = '../data/test_images/'

vidcap = cv2.VideoCapture('../data/test.mp4')
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite(os.path.join(images ,"frame%d.jpg" % count), image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

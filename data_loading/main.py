import numpy as np
import os
import glob
from cv2 import cv2

imagesPath = '/media/Data/CK+/CK+/cohn-kanade-images'

allImageData = {}

for subject in os.listdir(imagesPath):
    subjectPath = os.path.join(imagesPath,subject)
    if os.path.isdir(subjectPath):
        subjectData = {}
        for sequence in os.listdir(subjectPath):
            sequencePath = os.path.join(subjectPath,sequence)
            if os.path.isdir(sequencePath):
                print(sequencePath)
                imagePaths = []
                for sequenceFile in os.listdir(sequencePath):
                    if sequenceFile.endswith('.png'):
                        path = os.path.join(sequencePath, sequenceFile)
                        imagePaths.append(path)
                imagePaths.sort()
                images = []

                for path in imagePaths:
                    image = cv2.imread(path)
                    height, width, channels = image.shape
                    images.append(image)

                sequenceImages = np.stack(images)
                #print(sequenceImages.shape)

                subjectData[sequence] = sequenceImages
        allImageData[subject] = subjectData


emotionsPath = '/media/Data/CK+/CK+/Emotion'
allEmotionData = {}

for subject in os.listdir(emotionsPath):
    subjectPath = os.path.join(emotionsPath,subject)
    if os.path.isdir(subjectPath):
        subjectData = {}
        for sequence in os.listdir(subjectPath):
            sequencePath = os.path.join(subjectPath,sequence)
            if os.path.isdir(sequencePath):
                emotionLabels = []
                for sequenceFile in os.listdir(sequencePath):
                    path = os.path.join(sequencePath, sequenceFile)
                    emotionFile = open(path, "r")
                    emotionLabel = emotionFile.read()
                    if(emotionLabel != ""):
                        print(sequencePath)
                        print(emotionLabel)
                        emotionLabels.append(emotionLabel)
                subjectData[sequence] = emotionLabels
        allEmotionData[subject] = subjectData

print(allImageData["S005"]["001"])
print(allEmotionData["S005"]["001"])


import numpy as np
import os
import glob
from cv2 import cv2

pathLittleBear = '/media/Data/CK+/CK+/cohn-kanade-images'

pathLocal = r'c:\CK+'

allData = {}

for subject in os.listdir(pathLocal):
    subjectPath = os.path.join(pathLocal,subject)
    if os.path.isdir(subjectPath):
        subjectData = {}
        for sequence in os.listdir(subjectPath):
            sequencePath = os.path.join(subjectPath,sequence)
            if os.path.isdir(sequencePath):
                imagePaths = []
                for sequenceFile in os.listdir(sequencePath):
                    if sequenceFile.endswith('.png'):
                        path = os.path.join(sequencePath, sequenceFile)
                        imagePaths.append(path)
                imagePaths.sort()
                images = []

                for path in imagePaths:
                    image = cv2.imread(path)
                    images.append(image)

                #create numpy array of images
                sequenceImages = np.zeros(len(images))
                sequenceImages = images
                #todo put the images in a numpy stack
                #todo define shape of matrix

                subjectData[sequence] = sequenceImages
        allData[subject] = subjectData

print(allData)

        

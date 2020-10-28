import numpy as np
import os
import glob
from cv2 import cv2

class SequenceData:
    frames = []
    emotionLabel = ""
    facsLabels = ""
    landmarks = []

    def __init__(self):
        pass

#todo create a map of image data, emotion, and facs 

ckPath = '/mnt/Data/CK+/CK+/'
imagesPath = ckPath + 'cohn-kanade-images'
emotionsPath = ckPath + 'Emotion'
facsPath = ckPath + 'FACS'
landmarksPath = ckPath + 'Landmarks'

CKData = {}

imageData = {}
emotionData = {}
facsData = {}

#TODO other methods rely on image data to load subjects, pull that out into its own method probably
def getImageData():
    for subject in os.listdir(imagesPath):
        subjectPath = os.path.join(imagesPath,subject)
        if os.path.isdir(subjectPath):
            subjectSequences = {}
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    # print(sequencePath)
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
                    sequenceFrames = np.stack(images)
                    #print(sequenceImages.shape)
                    sequenceData = SequenceData()
                    sequenceData.frames = sequenceFrames
                    subjectSequences[sequence] = sequenceData
            CKData[subject] = subjectSequences

def getEmotionData():
    for subject in os.listdir(emotionsPath):
        subjectPath = os.path.join(emotionsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    emotionLabels = []
                    for sequenceFile in os.listdir(sequencePath):
                        path = os.path.join(sequencePath, sequenceFile)
                        emotionFile = open(path, "r")
                        emotionLabel = emotionFile.read()
                        if(emotionLabel != ""):
                            # print(sequencePath)
                            # print(emotionLabel)
                            emotionLabels.append(emotionLabel)
                    CKData[subject][sequence].emotionLabel = emotionLabel

def getFacsData():
    for subject in os.listdir(facsPath):
        subjectPath = os.path.join(facsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    facsLabels = []
                    for sequenceFile in os.listdir(sequencePath):
                        path = os.path.join(sequencePath, sequenceFile)
                        facsFile = open(path, "r")
                        facsLabel = facsFile.readlines()
                        if(facsLabel != ""):
                            # print(sequencePath)
                            # print(facsLabel)
                            facsLabels.append(facsLabel)
                    CKData[subject][sequence].facsLabels = facsLabels

#TODO preallocate for landmarks, constant length
def getLandmarksData():
    for subject in os.listdir(landmarksPath):
        subjectPath = os.path.join(landmarksPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    # print(sequencePath)
                    landmarks = []
                    for sequenceFile in os.listdir(sequencePath):
                        if sequenceFile.endswith('.txt'):
                            path = os.path.join(sequencePath, sequenceFile)
                            landmarksFile = open(path, "r")
                            landmark = landmarksFile.readlines()
                            if(landmark != ""):
                                print(path)
                                print(landmark)
                                landmarks.append(landmark)
                    CKData[subject][sequence].landmarks.append(landmarks)

                    

getImageData()
# getEmotionData()
# getFacsData()
getLandmarksData()

print(CKData["S005"]["001"].landmarks[0])
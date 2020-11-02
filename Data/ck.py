import numpy as np
import os
import glob
from cv2 import cv2

class SequenceData:
    frames = []
    emotionLabel = ""
    facsLabels = ""
    #todo needs to become a numpy array
    landmarks = []

    def __init__(self):
        pass

#todo create a map of image data, emotion, and facs 

ckPath = '/mnt/Data/CK+/CK+/'
imagesPath = ckPath + 'cohn-kanade-images'
emotionsPath = ckPath + 'Emotion'
facsPath = ckPath + 'FACS'
landmarksPath = ckPath + 'Landmarks'


#TODO other methods rely on image data to load subjects, pull that out into its own method probably
def getImageData(CKData):
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

def getEmotionData(CKData):
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

def getFacsData(CKData):
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


def readLandmarks(path):
    #todo fix this shape later
    landmarks = np.zeros(136)
    with open(path, "r") as landmarksFile:
        index = 0
        for line in landmarksFile:
            for point in line.split():
                if(point != "" and index < 136):
                    landmarks[index] = point
                    if(index == 0):
                        print(path)
                        print(point)
                        print(landmarks[index])
                    index += 1


def getLandmarksData(CKData):
    for subject in os.listdir(landmarksPath):
        subjectPath = os.path.join(landmarksPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    # print(sequencePath)
                    for sequenceFile in sorted(os.listdir(sequencePath)):
                        if sequenceFile.endswith('.txt'):
                            path = os.path.join(sequencePath, sequenceFile)
                            landmarks = readLandmarks(path)
                            CKData[subject][sequence].landmarks.append(landmarks)
                    CKData[subject][sequence].landmarks = np.array(CKData[subject][sequence].landmarks)


def main():

    CKData = {}
    getImageData(CKData)
    # getEmotionData(CKData)
    # getFacsData(CKData)
    getLandmarksData(CKData)

    print(CKData["S005"]["001"].landmarks[0])

if __name__ == "__main__":
    main()



import numpy as np
import os
import glob
from cv2 import cv2
from typing import Tuple

class SequenceData:
    frames = []
    emotionLabel = ""
    facsLabels = ""
    landmarks = []

    def __init__(self):
        pass

ckPath = '/mnt/Data/CK+/CK+/'
imagesPath = ckPath + 'cohn-kanade-images'
emotionsPath = ckPath + 'Emotion'
facsPath = ckPath + 'FACS'
landmarksPath = ckPath + 'Landmarks'

#todo fix storage of emotion data?
def getEmotionData():
    emotionData = []
    for subject in sorted(os.listdir(emotionsPath)):
        subjectPath = os.path.join(emotionsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in sorted(os.listdir(subjectPath)):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    emotionLabels = []
                    for sequenceFile in sorted(os.listdir(sequencePath)):
                        path = os.path.join(sequencePath, sequenceFile)
                        emotionFile = open(path, "r")
                        emotionLabel = emotionFile.read()
                        if(emotionLabel != ""):
                            #print(sequencePath)
                            #print(emotionLabel)
                            emotionLabels.append(emotionLabel)
                    emotionData.append(emotionLabel)
    return np.asarray(emotionData)

def getFacsDataWithoutIntensity():
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    presentActionUnits = []
    facsLabels = []
    for subject in sorted(os.listdir(facsPath)):
        subjectPath = os.path.join(facsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in os.listdir(subjectPath):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    allFacsLabels = []
                    for sequenceFile in os.listdir(sequencePath):
                        sequenceFacsLabels = np.zeros(shape=30)
                        path = os.path.join(sequencePath, sequenceFile)
                        facsFile = open(path, "r")
                        index = 0
                        for line in facsFile:
                            for i, sequenceActionUnit in enumerate(line.split()):
                                sequenceActionUnitInt = float(sequenceActionUnit)
                                j = 0
                                for au in allActionUnits:
                                    #skip the intensity
                                    if i == 0:
                                        if sequenceActionUnitInt == au:
                                            sequenceFacsLabels[j] = 1
                                            presentActionUnits.append(au)
                                    j+=1
                        allFacsLabels.append(sequenceFacsLabels)
                    facsLabels.append(allFacsLabels)
    return np.asarray(facsLabels, dtype=np.uint8)

def getSubjectsAndPeakEmotionFrame(target_shape: Tuple[int, int], make_square: bool):
    subjects = []
    for subject in sorted(os.listdir(imagesPath)):
        subjectPath = os.path.join(imagesPath,subject)
        subjects.append(subject)
        if os.path.isdir(subjectPath):
            subjectFinalImages = []
            for sequence in sorted(os.listdir(subjectPath)):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    # print(sequencePath)
                    imagePaths = []
                    for sequenceFile in sorted(os.listdir(sequencePath)):
                        if sequenceFile.endswith('.png'):
                            path = os.path.join(sequencePath, sequenceFile)
                            imagePaths.append(path)
                    lastImage = cv2.imread(imagePaths[-1])

                    if make_square:
                        height, width = lastImage.shape[:2]
                        trim_size = (width - height) // 2
                        lastImage = lastImage[:, trim_size:height+trim_size]

                    if(target_shape is not None):
                        lastImage = cv2.resize(lastImage, target_shape)

                    subjectFinalImages.append(lastImage)
    return np.asarray(subjects), np.asarray(subjectFinalImages)

def getLastFrameData(target_shape: Tuple[int, int], make_square: bool):
    subjects, peakEmotionImages = getSubjectsAndPeakEmotionFrame(target_shape, make_square)
    return subjects, \
           peakEmotionImages, \
           getEmotionData(), \
           getFacsDataWithoutIntensity()


def main():
    subjects, lastFrameImages, emotionData, facs = getLastFrameData((256, 256), True)
    print(subjects)

if __name__ == "__main__":
    main()


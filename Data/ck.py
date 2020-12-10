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
                            emotionLabels.append(emotionLabel)
                    emotionData.append(emotionLabel)
    return np.asarray(emotionData)


#parameter: selectedActionUnits is an optional list of the desired action units to track.
#if no list is passed in, allActionUnits will be selected
def getLastFrameFacsDataWithoutIntensity(selectedActionUnits = []):
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    if not selectedActionUnits:
        selectedActionUnits = allActionUnits
    sumOfActionUnits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    presentActionUnits = []
    facsLabels = []
    for subject in sorted(os.listdir(facsPath)):
        subjectPath = os.path.join(facsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in sorted(os.listdir(subjectPath)):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    allFacsLabels = []
                    facsFilePaths = []
                    for facsFile in sorted(os.listdir(sequencePath)):
                        path = os.path.join(sequencePath, facsFile)
                        facsFilePaths.append(path)
                    sequenceFacsLabels = np.zeros(shape=30)
                    facsFile = open(facsFilePaths[-1], "r")
                    for line in facsFile:
                        for i, sequenceActionUnit in enumerate(line.split()):
                            sequenceActionUnitInt = float(sequenceActionUnit)
                            if sequenceActionUnitInt in selectedActionUnits:
                                j = 0
                                for au in allActionUnits:
                                    #skip the intensity
                                    if i == 0:
                                        if sequenceActionUnitInt == au:
                                            sequenceFacsLabels[j] = 1
                                            sumOfActionUnits[j] += 1
                                            presentActionUnits.append(au)
                                    j+=1
                        allFacsLabels.append(sequenceFacsLabels)
                    facsLabels.append(allFacsLabels)
    return np.asarray(facsLabels), np.asarray(sumOfActionUnits)

def get_aus_with_n_pct_positive(sumOfActionUnits,minimumFrequency):
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    totalNumberofSequences = 593
    viableActionUnits = []
    for i,auSum in enumerate(sumOfActionUnits):
        pctPositive = auSum/totalNumberofSequences
        if  pctPositive >= minimumFrequency:
            viableActionUnits.append(allActionUnits[i])
    return np.asarray(viableActionUnits), np.searchsorted(allActionUnits, viableActionUnits)

def readLandmarks(path):
    landmarks = np.zeros(shape=(68,2))
    with open(path, "r") as landmarksFile:
        i = 0
        for line in landmarksFile:
            j = 0
            for point in line.split():
                if(point != ""):
                    #print(path)
                    #print(point)
                    landmarks[i][j] = point
                    j+=1
            i+=1
    return landmarks

def getLastFrames(target_shape: Tuple[int, int], make_square: bool):
    subjectSequenceImages = []
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

                    print(imagePaths[-1])
                    subjectFinalImages.append(lastImage)
            subjectSequenceImages.append(subjectFinalImages)
    return np.asarray(subjects), np.asarray(subjectSequenceImages)

#TODO fix storage of emotion data?
def getLastFrameData(target_shape: Tuple[int, int], make_square: bool):
    subjects, subjectSequenceImages = getLastFrames((256, 256), True)
    facs, sumOfActionUnits = getLastFrameFacsDataWithoutIntensity()
    return subjects, subjectSequenceImages, getEmotionData(), facs, sumOfActionUnits

def main():
    subjects, lastFrameImages, emotionData, facs, sumOfActionUnits = getLastFrameData((256, 256), True)
    viableAUs = get_aus_with_n_pct_positive(sumOfActionUnits, 0.0)

if __name__ == "__main__":
    main()

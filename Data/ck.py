import numpy as np
import os
import glob
from cv2 import cv2
from typing import Tuple

ckPath = '/mnt/Data/CK+/CK+/'
imagesPath = ckPath + 'cohn-kanade-images'
emotionsPath = ckPath + 'Emotion'
facsPath = ckPath + 'FACS'
landmarksPath = ckPath + 'Landmarks'

"""
@return A numpy array of the string emotion data from each sequence
"""
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

"""
@selectedActionUnits: An optional list of the desired action units to track. 
    If no list is passed in, allActionUnits will be selected.
@return Two numpy arrays: one for the facslabels from each sequence and one for 
    the total sum of each action unit when present. 
"""
def getFacsData(selectedActionUnits = []):
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    if not selectedActionUnits:
        selectedActionUnits = allActionUnits
    sumOfActionUnits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    intensityOfActionUnits = {}
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
                            #get the au
                            if i == 0:
                                sequenceActionUnitInt = float(sequenceActionUnit)
                                if sequenceActionUnitInt in selectedActionUnits:
                                    j = 0
                                    for au in allActionUnits:
                                        if sequenceActionUnitInt == au:
                                            sequenceFacsLabels[j] = 1
                                            sumOfActionUnits[j] += 1
                                            intensityOfActionUnits[au] = 0
                                        j+=1
                            #get the intensity
                            if i == 1:
                                intensityOfActionUnits[au] = 0
                    facsLabels.append(sequenceFacsLabels)
    return np.asarray(facsLabels), np.asarray(sumOfActionUnits)

"""
@selectedActionUnits: An optional list of the desired action units to track. 
    If no list is passed in, allActionUnits will be selected.
@return Two numpy arrays: one for the facslabels from each sequence and one for 
    the total sum of each action unit when present. 
"""
def getFacsDataWithoutIntensity(selectedActionUnits = []):
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
"""
@sumOfActionUnits: A list of the total sum of each action unit when present. 
@minimumFrequency: A decimal representation of a percentage that represents 
    the minimum percentage of occurences for an AU to be considered viable.
@return Two numpy arrays: One for the action units that have been determined 
    to be viable and one for the indices of said action units.
"""
def get_aus_with_n_pct_positive(sumOfActionUnits,minimumFrequency):
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    totalNumberofSequences = 593
    viableActionUnits = []
    for i,auSum in enumerate(sumOfActionUnits):
        pctPositive = auSum/totalNumberofSequences
        if  pctPositive >= minimumFrequency:
            viableActionUnits.append(allActionUnits[i])
    return np.asarray(viableActionUnits), np.searchsorted(allActionUnits, viableActionUnits)

"""
@path: A string path to the landmarks file.
@returns: A numpy array of the landmarks from this file.
"""
def readLandmarks(path):
    landmarks = np.zeros(shape=(68,2))
    with open(path, "r") as landmarksFile:
        i = 0
        for line in landmarksFile:
            j = 0
            for point in line.split():
                if(point != ""):
                    landmarks[i][j] = point
                    j+=1
            i+=1
    return landmarks

"""
@returns A list of all landmark data.
"""
def getLandmarksData():
    landmarksList = []
    for subject in sorted(os.listdir(landmarksPath)):
        subjectPath = os.path.join(landmarksPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in sorted(os.listdir(subjectPath)):
                seqDict = {}
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    sequenceLandmarkList = []
                    for sequenceFile in sorted(os.listdir(sequencePath)):
                        if sequenceFile.endswith('.txt'):
                            path = os.path.join(sequencePath, sequenceFile)
                            landmarks = readLandmarks(path)
                            sequenceLandmarkList.append(landmarks)
                    seqDict[sequence] = sequenceLandmarkList
                landmarksList.append(seqDict)
    return landmarksList

"""
@target_shape: A Tuple of ints that determines the shape of the image.
@make_square: A boolean that determines if the image will be square.
@returns: A numpy array of subject names and a numpy array of the sequence images.
"""
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

"""
@target_shape: A Tuple of ints that determines the shape of the image.
@make_square: A boolean that determines if the image will be square.
@returns: a numpy array of subject names, sequence images, emotions, facs data, and the total sum of each action unit when present. 
"""
def getLastFrameData(target_shape: Tuple[int, int], make_square: bool):
    subjects, subjectSequenceImages = getLastFrames((256, 256), True)
    facs, sumOfActionUnits = getFacsDataWithoutIntensity()
    return subjects, subjectSequenceImages, getEmotionData(), facs, sumOfActionUnits

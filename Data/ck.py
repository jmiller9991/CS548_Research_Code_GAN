import numpy as np
import os
import glob
from cv2 import cv2
from typing import Tuple

"""
@cKPath: A string path to the CK+ Database
@rtype numpy array of strings
@return The emotion data from each sequence
"""
def getEmotionData(ckPath):
    emotionsPath = ckPath + 'Emotion'
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
@cKPath: A string path to the CK+ Database
@selectedActionUnits: An optional list of the desired action units to track. 
    If no list is passed in, allActionUnits will be selected.
@rtype nested numpy arrays of ints, numpy array of ints, nested numpy array of ints   
@return The facslabels from each sequence, the total sum of each action unit when present and the intensity of each AU. 
"""
def getFacsData(ckPath,selectedActionUnits = []):
    facsPath = ckPath + 'FACS'
    allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
    if not selectedActionUnits:
        selectedActionUnits = allActionUnits
    sumOfActionUnits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    actionUnitIntensities = []
    facsLabels = []
    for subject in sorted(os.listdir(facsPath)):
        subjectPath = os.path.join(facsPath,subject)
        if os.path.isdir(subjectPath):
            for sequence in sorted(os.listdir(subjectPath)):
                sequencePath = os.path.join(subjectPath,sequence)
                if os.path.isdir(sequencePath):
                    intensityOfActionUnits = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    facsFilePaths = []
                    for facsFile in sorted(os.listdir(sequencePath)):
                        path = os.path.join(sequencePath, facsFile)
                        facsFilePaths.append(path)
                    sequenceFacsLabels = np.zeros(shape=30)
                    facsFile = open(facsFilePaths[-1], "r")
                    for line in facsFile:
                        indexOfPresentAu = -1
                        for i, item in enumerate(line.split()):
                            #get the au
                            if i == 0:
                                sequenceActionUnitInt = float(item)
                                if sequenceActionUnitInt in selectedActionUnits:
                                    j = 0
                                    for au in allActionUnits:
                                        if sequenceActionUnitInt == au:
                                            sequenceFacsLabels[j] = 1
                                            sumOfActionUnits[j] += 1
                                            indexOfPresentAu = j
                                        j+=1
                            #get the intensity
                            if i == 1:
                                if indexOfPresentAu >= 0:
                                    intensityOfActionUnits[indexOfPresentAu] = item 
                    facsLabels.append(sequenceFacsLabels)
                    actionUnitIntensities.append(intensityOfActionUnits)
    return np.asarray(facsLabels), np.asarray(sumOfActionUnits), np.asarray(actionUnitIntensities)

"""
@cKPath: A string path to the CK+ Database
@selectedActionUnits: An optional list of the desired action units to track. 
    If no list is passed in, allActionUnits will be selected.
@rtype nested numpy arrays of ints, numpy array of ints  
@return The facslabels from each sequence and the total sum of each action unit when present. 
"""
def getFacsDataWithoutIntensity(ckPath, selectedActionUnits = []):
    facsPath = ckPath + 'FACS'
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
@rtype numpy array of ints, numpy array of ints
@return The action units that have been determined 
    to be viable and the indices of said action units.
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
@rtype: numpy array of strings
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
@cKPath: A string path to the CK+ Database
@rtype nested lists of strings
@returns A list of all landmark data.
"""
def getLandmarksData(ckPath):
    landmarksPath = ckPath + 'Landmarks'
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
@cKPath: A string path to the CK+ Database
@target_shape: A Tuple of ints that determines the shape of the image.
@make_square: A boolean that determines if the image will be square.
@rtype: numpy array of strings, numpy array of image data
@returns: Subject names and sequence image data.
"""
def getLastFrames(ckPath,target_shape: Tuple[int, int], make_square: bool):
    imagesPath = ckPath + 'cohn-kanade-images'
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
@cKPath: A string path to the CK+ Database
@target_shape: A Tuple of ints that determines the shape of the image.
@make_square: A boolean that determines if the image will be square.
@rtype: numpy array of strings, numpy array of images, nested numpy array of ints, nested numpy array of ints, numpy array of ints 
@returns: Subject names, sequence images, emotions, facs data, and the total sum of each action unit when present. 
"""
def getLastFrameData(ckPath, target_shape: Tuple[int, int], make_square: bool):
    subjects, subjectSequenceImages = getLastFrames(ckPath, (256, 256), True)
    facs, sumOfActionUnits = getFacsDataWithoutIntensity(ckPath)
    return subjects, subjectSequenceImages, getEmotionData(ckPath), facs, sumOfActionUnits
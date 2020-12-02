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
    return np.array(emotionData)

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
                        # print(facsFile)
                        # print(sequenceFacsLabels)
                        allFacsLabels.append(sequenceFacsLabels)
                    facsLabels.append(allFacsLabels)
    return np.asarray(facsLabels), np.asarray(sumOfActionUnits)

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

def getLastFrames():
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
                    print(imagePaths[-1])
                    subjectFinalImages.append(lastImage)
            subjectSequenceImages.append(subjectFinalImages)
    return np.array(subjects), np.asarray(subjectSequenceImages)

#todo fix storage of emotion data?
def getLastFrameData():
    subjects, subjectSequenceImages = getLastFrames()
    facs, sumOfActionUnits = getLastFrameFacsDataWithoutIntensity()
    return subjects, subjectSequenceImages, getEmotionData(), facs, sumOfActionUnits

def main():
    subjects, lastFrameImages, emotionData, facs, sumOfActionUnits = getLastFrameData()
    print(sumOfActionUnits)
if __name__ == "__main__":
    main()

#TODO
# make sure intensity is stored somewhere

# pull out separate presesnt AUs into their own array 

# create getLastFrameData
# images, ground, subjects = ck.getLastFrameData()
# will be one numpy array. returns 3 numpy arrays: 
# images: (samples, height, width, channels)

# ground: (samples, numClasses)

# subjects: (samples)

# np sum axis 0 that is # aus X # 



#TODO other methods rely on image data to load subjects, pull that out into its own method probably
# def getImageData(CKData):
#     for subject in os.listdir(imagesPath):
#         subjectPath = os.path.join(imagesPath,subject)
#         if os.path.isdir(subjectPath):
#             subjectSequences = {}
#             for sequence in os.listdir(subjectPath):
#                 sequencePath = os.path.join(subjectPath,sequence)
#                 if os.path.isdir(sequencePath):
#                     # print(sequencePath)
#                     imagePaths = []
#                     for sequenceFile in os.listdir(sequencePath):
#                         if sequenceFile.endswith('.png'):
#                             path = os.path.join(sequencePath, sequenceFile)
#                             imagePaths.append(path)
#                     imagePaths.sort()
#                     images = []
#                     for path in imagePaths:
#                         image = cv2.imread(path)
#                         height, width, channels = image.shape
#                         images.append(image)
#                     sequenceFrames = np.stack(images)
#                     #print(sequenceImages.shape)
#                     sequenceData = SequenceData()
#                     sequenceData.frames = sequenceFrames
#                     subjectSequences[sequence] = sequenceData
#             CKData[subject] = subjectSequences

# def getFacsData(CKData):
#     for subject in os.listdir(facsPath):
#         subjectPath = os.path.join(facsPath,subject)
#         if os.path.isdir(subjectPath):
#             for sequence in os.listdir(subjectPath):
#                 sequencePath = os.path.join(subjectPath,sequence)
#                 if os.path.isdir(sequencePath):
#                     facsLabels = []
#                     for sequenceFile in os.listdir(sequencePath):
#                         path = os.path.join(sequencePath, sequenceFile)
#                         facsFile = open(path, "r")
#                         facsLabel = facsFile.readlines()
#                         if(facsLabel != ""):
#                             # print(sequencePath)
#                             # print(facsLabel)
#                             facsLabels.append(facsLabel)
#                     CKData[subject][sequence].facsLabels = facsLabels

# def getFacsDataWithoutIntensity():
#     allActionUnits = np.array([1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,31,34,38,39,43])
#     presentActionUnits = []
#     facsLabels = []
#     for subject in sorted(os.listdir(facsPath)):
#         subjectPath = os.path.join(facsPath,subject)
#         if os.path.isdir(subjectPath):
#             for sequence in os.listdir(subjectPath):
#                 sequencePath = os.path.join(subjectPath,sequence)
#                 if os.path.isdir(sequencePath):
#                     allFacsLabels = []
#                     for sequenceFile in os.listdir(sequencePath):
#                         sequenceFacsLabels = np.zeros(shape=30)
#                         path = os.path.join(sequencePath, sequenceFile)
#                         facsFile = open(path, "r")
#                         index = 0
#                         for line in facsFile:
#                             for i, sequenceActionUnit in enumerate(line.split()):
#                                 sequenceActionUnitInt = float(sequenceActionUnit)
#                                 j = 0
#                                 for au in allActionUnits:
#                                     #skip the intensity
#                                     if i == 0:
#                                         if sequenceActionUnitInt == au:
#                                             sequenceFacsLabels[j] = 1
#                                             presentActionUnits.append(au)
#                                     j+=1
#                         print(sequenceFile)
#                         print(sequenceFacsLabels)
#                         allFacsLabels.append(sequenceFacsLabels)
#                     facsLabels.append(allFacsLabels)
#                     # CKData[subject][sequence].facsLabels = allFacsLabels
#     return np.array(facsLabels)

# def getLandmarksData(CKData):
#     for subject in os.listdir(landmarksPath):
#         subjectPath = os.path.join(landmarksPath,subject)
#         if os.path.isdir(subjectPath):
#             for sequence in os.listdir(subjectPath):
#                 sequencePath = os.path.join(subjectPath,sequence)
#                 if os.path.isdir(sequencePath):
#                     # print(sequencePath)
#                     for sequenceFile in sorted(os.listdir(sequencePath)):
#                         if sequenceFile.endswith('.txt'):
#                             path = os.path.join(sequencePath, sequenceFile)
#                             landmarks = readLandmarks(path)
#                             CKData[subject][sequence].landmarks.append(landmarks)
#                     CKData[subject][sequence].landmarks = np.array(CKData[subject][sequence].landmarks)



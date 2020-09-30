import numpy as np
import os
import glob
from cv2 import cv2



#for each subject, for each expression, get all images



pathLittleBear = '/media/Data/CK+/CK+/cohn-kanade-images'

pathLocal = r'c:\Images'

allData = {}

for subject in os.listdir(pathLocal):
    subjectPath = os.path.join(pathLocal,subject)
    if os.path.isdir(subjectPath):
        subjectData = {}
        for sequence in os.listdir(subjectPath):
            sequencePath = os.path.join(subjectPath,sequence)
            if os.path.isdir(sequencePath):
                pngPaths = []
                for sequenceFile in os.listdir(sequencePath):
                    if sequenceFile.endswith('.png'):
                        pngPaths.append(os.path.join(sequencePath, sequenceFile))
                pngPaths.sort()
                #create numpy array of the actual images here, use open cv

                subjectData[sequence] = pngPaths
                #todo 
                #subjectData[sequence] = pngData(this represents the actual numpy array)
        allData[subject] = subjectData

print(allData)

        

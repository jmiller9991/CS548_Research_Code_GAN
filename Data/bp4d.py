# MIT LICENSE
#
# Copyright 2020 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import sys
import numpy as np
import imageio
from Data.Common import *
#from Data.Common3D import *
#from Data.BP4D.BP4DFeaturePoints3D import *

allSubjects_BP4D = [
    'F001',
	'F002',
	'F003',
	'F004',
	'F005',
	'F006',
	'F007',
	'F008',
	'F009',
	'F010',
	'F011',
	'F012',
	'F013',
	'F014',
	'F015',
	'F016',
	'F017',
	'F018',
	'F019',
	'F020',
	'F021',
	'F022',
	'F023',
	'M001',
	'M002',
	'M003',
	'M004',
	'M005',
	'M006',
	'M007',
	'M008',
	'M009',
	'M010',
	'M011',
	'M012',
	'M013',
	'M014',
	'M015',
	'M016',
	'M017',
	'M018'
    ]
				
allTasks_BP4D = [	
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8"
    ]

DRMLOccAUs_BP = ["AU01",
				 "AU02",
				 "AU04",
				 "AU06",
				 "AU07",
				 "AU10",
				 "AU12",
				 "AU14",
				 "AU15",
				 "AU17",
				 "AU23",
				 "AU24"
				 ]

DATA_AU_INT = "AU_INT"
DATA_AU_OCC = "AU_OCC"
DATA_PAIN = "PAIN"

DATA_3D = "DATA_3D"
DATA_IMAGE = "DATA_IMAGE"

# Max pain intensity (AU4 + AU6 + AU10)
MAX_PAIN_INTENSITY = 15.0

# Max AU intensity
MAX_AU_INTENSITY = 5.0

def load_ground(dataDir, 
                groundDir, 
                groundType, 
                groundRequested,
                lostListFilename=None, 
                subjectsRequested=allSubjects_BP4D, 
                tasksRequested=allTasks_BP4D,                
                facialFeaturePointPath=None,
                normalizeLabels=False,                
                DEBUG_PRINT=True):

    # Check to make sure we have more than one class requested
    if len(groundRequested) < 1:
        raise ValueError("Must request at least ONE item in groundRequested!")

    # Read in list of lost files (if available)
    if lostListFilename is not None:
        with open(lostListFilename) as lostFile:
            lostFiles = lostFile.readlines()
            lostFiles = [x.rstrip("\n") for x in lostFiles]
            if DEBUG_PRINT: print("Lost file count:", len(lostFiles))
    else:
        lostFiles = []

    # Get full path to ground truth folder
    startGroundFolder = join(groundDir, groundType)

    # Are we loading feature points?
    doLoadFeaturePoints = facialFeaturePointPath is not None

    # Set file extensions
    fileExtension = ".obj"
    featureExtension = ".tbnd"

    # Prepare lists for loading
    allLabels = []
    allSubjects = []
    allTasks = []
    allFilenames = []
    allFeaturePaths = []

    # For each requested subject...
    if DEBUG_PRINT: print("Loading filenames and ground truth...")    
    for subject in subjectsRequested:
        # Get path to image subject folder
        currentSubjectPath = join(dataDir, subject)
        if DEBUG_PRINT: print("\t", currentSubjectPath)

        # For each requested task...
        for task in tasksRequested:	

            # STEP ONE
            # Initialize the filename and labels
            taskFilenames = []
            taskFeaturePaths = []
            taskLabels = []
            taskGoodFrame = []

            # Figure out correct ground filename
            currentClass = groundRequested[0]
            if groundType == DATA_AU_INT or groundType == DATA_PAIN:
                labelfilename = subject + "_" + task + "_" + currentClass + ".csv"
            elif groundType == DATA_AU_OCC:
                labelfilename = subject + "_" + task + ".csv"
            else:
                raise ValueError("Invalid groundType:", groundType)

            # Figure out correct folder (INT vs OCC/PAIN)
            if groundType == DATA_AU_INT:
                finalGroundFolder = join(startGroundFolder, currentClass)
            else:
                finalGroundFolder = startGroundFolder

            # Figure out zero padding            
            digitCnt = getZeroPadNumber(join(currentSubjectPath, task), fileExtension)

            # Open file to:
            # - Create label placeholders (all zeros initially)
            # - Add "image" filenames you need
            # NOTE: 
            # OCC 		--> first row has label
            # INT/PAIN 	--> first row is real data
            with open(join(finalGroundFolder, labelfilename), 'rt') as csvfile:
                csvReader = csv.reader(csvfile, delimiter=",")
                firstRow = True
                for row in csvReader:
                    if groundType == DATA_AU_OCC:
                        if firstRow:
                            firstRow = False
                            continue

                    # Create placeholder for labels
                    perFrameLabels = list(np.zeros((len(groundRequested))))
                    taskLabels.append(perFrameLabels)

                    # Get frame
                    index = row[0]							
                    index = padZero(index, digitCnt)

                    # Get filename for image
                    fileName = index 

                    # Get file path
                    filePath = join(subject, join(task, fileName))
                    
                    # Append to list of filenames
                    taskFilenames.append(join(dataDir, filePath + fileExtension))
                    
                    # Check if good frame                                            
                    taskGoodFrame.append(filePath not in lostFiles)

                    # Add path to feature points 
                    if doLoadFeaturePoints:
                        featurePath = join(facialFeaturePointPath, 
                                        join(subject, 
                                            join(task, fileName + featureExtension)))
                        taskFeaturePaths.append(featurePath)
                    else:
                        taskFeaturePaths.append("")

                    # NOT first row anymore
                    firstRow = False
            
            # STEP TWO
            # For each possible class, get appropriate labels
            groundIndex = 0
            for currentClass in groundRequested:

                # Get filename and appropriate column
                if groundType == DATA_AU_INT:
                    filename = currentClass + "/" + subject + "_" + task + "_" + currentClass + ".csv"
                    MAX_VALUE = MAX_AU_INTENSITY
                    auColIndex = 1
                elif groundType == DATA_AU_OCC:
                    filename = subject + "_" + task + ".csv"
                    MAX_VALUE = 1.0
                    auColIndex = int(currentClass.split("AU")[1])
                elif groundType == DATA_PAIN:
                    filename = subject + "_" + task + "_" + currentClass + ".csv"
                    MAX_VALUE = MAX_PAIN_INTENSITY
                    auColIndex = 1

                # Open the actual file
                with open(join(startGroundFolder, filename), 'rt') as csvfile:
                    csvReader = csv.reader(csvfile, delimiter=",")
                    rowIndex = 0
                    firstRow = True

                    for row in csvReader:
                        # Skip first row if OCC
                        if groundType == DATA_AU_OCC:
                            if firstRow:
                                firstRow = False
                                continue

                        # Get the actual label
                        label = row[auColIndex]

                        # Convert the label to float type
                        label = float(label)

                        # Store label
                        taskLabels[rowIndex][groundIndex] = label

                        # Increment row index
                        rowIndex += 1

                        # Not first row anymore
                        firstRow = False

                # Go to next ground label
                groundIndex += 1

            # STEP THREE
            # Clean up and possibly normalize labels
            
            classIndex = 0
            # Make sure the label is valid
            for frameIndex in range(len(taskLabels)):
                #print("\t", frameIndex)
                for classIndex in range(len(taskLabels[frameIndex])):
                    # Get int label
                    currentLabel = int(taskLabels[frameIndex][classIndex])
                    
                    # Is this INT or OCC?
                    if (groundType == DATA_AU_INT 
                        or groundType == DATA_AU_OCC):
                        # Checking if 9; if it is, reject whole frame						
                        if currentLabel > 7:
                            taskGoodFrame[frameIndex] = False
                        else:
                            # Otherwise, normalize labels if requested
                            if normalizeLabels:
                                taskLabels[frameIndex][classIndex] /= MAX_VALUE
                    elif groundType == DATA_PAIN:
                        # Checking if negative
                        if currentLabel < 0:
                            taskGoodFrame[frameIndex] = False
                        else:
                            # Otherwise, normalize
                            if normalizeLabels:
                                taskLabels[frameIndex][classIndex] /= MAX_VALUE

            # STEP FOUR
            # Split sequences if there are bad frames in between

            sequenceFilenames = []
            sequenceFeaturePaths = []
            sequenceLabels = []            
            wasBad = False

            oneSeqFilenames = []
            oneSeqFeaturePaths = []
            oneSeqLabels = []    
            
            for i in range(len(taskGoodFrame)):
                if taskGoodFrame[i]:
                    # Hit good frame

                    # Did we just finish a run of bad frames?
                    if wasBad:
                        # Add the sequence we were building so far and start a new one
                        if len(oneSeqFilenames) > 0:
                            sequenceFilenames.append(oneSeqFilenames)
                            sequenceFeaturePaths.append(oneSeqFeaturePaths)
                            sequenceLabels.append(oneSeqLabels)

                        oneSeqFilenames = []
                        oneSeqFeaturePaths = []
                        oneSeqLabels = []

                        # Not in a bad area anymore
                        wasBad = False
                    
                    # Add to our current lists
                    oneSeqFilenames.append(taskFilenames[i])
                    oneSeqFeaturePaths.append(taskFeaturePaths[i])
                    oneSeqLabels.append(taskLabels[i])

                else:
                    # Hit a bad area!
                    wasBad = True

            # Add last sequence
            if len(oneSeqFilenames) > 0:
                sequenceFilenames.append(oneSeqFilenames)
                sequenceFeaturePaths.append(oneSeqFeaturePaths)
                sequenceLabels.append(oneSeqLabels)
            else:
                print("WARNING: No sequences added for", subject,"/",task)

            # Concatenate with larger lists
            allFilenames += sequenceFilenames
            allFeaturePaths += sequenceFeaturePaths
            allLabels += sequenceLabels

            # Add subjects and tasks
                       
            for i in range(len(sequenceFilenames)):
                sequenceSubjects = []
                sequenceTasks = []
                for j in range(len(sequenceFilenames[i])):
                    sequenceSubjects.append(subject)
                    sequenceTasks.append(task)   
                allSubjects.append(sequenceSubjects)
                allTasks.append(sequenceTasks)


    # Add 2D extension
    imageExtension = ".jpg"

    allImageFilenames = []
    for sampleIndex in range(len(allFilenames)):
        imageSequence = []
        for frameIndex in range(len(allFilenames[sampleIndex])):
            # Get 3D path
            path3D = allFilenames[sampleIndex][frameIndex]
            # Remove and replace extension
            path = path3D[:-(len(fileExtension))]
            # Add 2D extension
            path2D = path + imageExtension
            # Add to sequence list
            imageSequence.append(path2D)
        # Add sequence to overall list
        allImageFilenames.append(imageSequence)

    # Set 3D paths
    all3DFilenames = allFilenames    

    if DEBUG_PRINT: 
        print("Ground truth loading complete.")

    # Return data
    if facialFeaturePointPath is not None:
        return all3DFilenames, allImageFilenames, allLabels, allSubjects, allTasks, allFeaturePaths 
    else:
        return all3DFilenames, allImageFilenames, allLabels, allSubjects, allTasks


def main():
    print("HELLO")


if __name__ == "__main__":
    main()


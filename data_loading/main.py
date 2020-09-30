import numpy as np
import os
import glob



pngFiles = []

pathLittleBear = '/media/Data/CK+/CK+/cohn-kanade-images/S005'

pathLocal = r'c:\Images\New Folder'

for root, dirs, files in os.walk(pathLittleBear):
    x=0
    for directory in dirs:
        if x == 2: 
            break
        x += 1
        for file in files:
            if file.endswith('.png'):
                pngFiles.append(os.path.join(root, file))

print(pngFiles)


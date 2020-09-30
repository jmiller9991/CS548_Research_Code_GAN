import numpy as np
import os
import glob



pngFiles = []

pathLittleBear = '/media/Data/CK+/CK+/cohn-kanade-images/S005'

pathLocal = r'c:\Images\New Folder'

for root, dirs, files in os.walk(pathLittleBear):
        for file in files:
            if file.endswith('.png'):
                pngFiles.append(os.path.join(root, file))

print(pngFiles)


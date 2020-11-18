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

import glob
import os
from os.path import join 

def getZeroPadNumber(folderPath, extension):	
    # Get iterator for files
    imageFiles = glob.iglob(join(folderPath, "*" + extension))

    # Grab first filename
    firstItem = next(imageFiles)
    firstItem = os.path.basename(firstItem)

    # Remove the extension
    itemLen = len(firstItem) - len(extension)
    
    # Get the maximum number of filenames
    maxItemCnt = pow(10, itemLen) - 1
    
    # Get digit count for zero padding
    digitCnt = getDigitCnt(maxItemCnt)
    
    return digitCnt

def getDigitCnt(cnt):
    digitCnt = 1
    while cnt >= 10:
        cnt /= 10
        digitCnt += 1
    return digitCnt


def padZero(index, digitCnt):
    curDigitCnt = getDigitCnt(int(index))
    diffDigit = digitCnt - curDigitCnt
    if diffDigit > 0:
        for i in range(diffDigit):
            index = "0" + index
    return index
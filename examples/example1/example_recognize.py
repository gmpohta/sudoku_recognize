import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import sudoku_recognize

'''
An example of how the class works. Load the sudoku screenshot "original.png" and call the class method "recognize_sudoku" to recognize it.
The output list "in_puzzle" contains the Sudoku numbers. The arrays coordx, coordy are the coordinates of the sudoku lines. 
'''

img = cv2.imread(sys.path[0] + "/original.png", 1)
recognizer=sudoku_recognize.Recognize_Sudoku()

in_puzzle,coordx,coordy=recognizer.recognize_sudoku(img)
print(in_puzzle)
cv2.waitKey(0)

import sys
import os
import cv2
path_current_dir=os.getcwd()
list_import=path_current_dir.split('\\')
sys.path.insert(0, '\\'.join(list_import[:-1]))
import sudoku_recognize

'''
An example of how the class works. Load the sudoku screenshot "original.png" and call the class method "recognize_sudoku" to recognize it.
The output list "in_puzzle" contains the Sudoku numbers. The arrays coordx, coordy are the coordinates of the sudoku lines. 
'''

image_file = "original.png"
img = cv2.imread(image_file)
recogn=sudoku_recognize.Recognize_Sudoku()

in_puzzle,coordx,coordy=recogn.recognize_sudoku(img)
print(in_puzzle)
cv2.waitKey(0)

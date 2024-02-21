 

## Description
This Kaggle is one part of a project from the course Artificial Neural Networks and 
Cognitive Models of the MAI program at THWS. The project objective is to train a deep 
learning model that can solve a Sudoku puzzle. To help you, we have split the problem into 
two tasks: in Task 1 you shall train a model to read the Sudoku puzzle to know which digits 
there are and where are their placed withing the Sudoku grid. In Task 2 you shall train a 
model to actually solve the Sudoku (using the standard rules), that is to complete the empty 
cells in the grid with the correct digits.


This competition is about Task 1 (here a link to Task 2)
For this competition, you shall train a model to read the Sudoku puzzle. The Sudoku puzzle 
is a 9x9 grid filled with digits from 1 to 9 following the Sudoku rules (each column, each 
row, and each 3x3 sub grid contains all of the digits). The partially completed grid that you 
shall be able to read here has always 40 cells empty. Your model shall be able to recognize 
the filled digits within the puzzle and their positions (cells) in the grid. In the end, it shall 
provide a list of all the cells with the digits within.
To train such a model we provide you with a set of 50000 training images (the partially 
solved Sudoku puzzles) and the corresponding labels (a list of coordinates of the 41 filled 
positions with the digits within for each train image). Your task is to recognize the digits and 
their positions for all 10000 test set images we provide.


## Evaluation
The goal of this task is to predict the correct label for each digit in all the test images. In 
addition to the digit class, you will also need to submit each digit's cell position in the puzzle.
The cell position within the 9x9 grid is described by its x (horizontal) and y (vertical) 
coordinates similarly as in a matrix. The indexes range from 0 to 8 such that top-left is (0, 0), 
top-right is (0, 8), bottom-left is (8, 0) and bottom-right is (8, 8).


## Submission File
Each test image is identified with an ID (0-9999). For each ID you will need to submit 81 
values (for each cell within the 9x9 grid). You shall use the value (digit) 0 for an empty cell. 
The submission file should contain a header and have the following format:
id,value
0_00,0
0_01,1
0_02,5
._.,.
._.,.
9999_88,9
Here id is composed of the ID of the image (0-9999) and the xy coordinate of the 
position ID_xy, value is the digit (1-9, with 0 used for empty positions). In total, there shall 
be 810001 lines in the submission file including the header.


## Dataset Description
**What files do I need?**
You need the training images with labels and test images.
**What should I expect the data format to be?**
You are provided with 3 NumPy files (.npy).
1. Training Images as NumPy array of size 50000x252x252.
2. Training labels as NumPy array of size 50000x41x3. There are 41 cells in the puzzle 
filled with digits placed at xy coordinates. These 3 variables are arranged as x,y,value.
3. Test images as NumPy array of size 10000x252x252.
**Were**
The data files are shared via a Google Drive link. You can either download the files OR 
mount the link to Google Collab and access them directly.
https://drive.google.com/drive/folders/1iyQDn_kE_QIgGvtxMRcBVqf_Gcxe9Pua?usp=drive_link
**What am I predicting?**
You need to predict the digits filled within the puzzle and their positions.
**What do I need to submit?**
You need to submit values for all 81 cells in the puzzle (for the 41 filled cells the recognized 
digit, for the empty cells use value zero) along with their x,y coordinates for all test images.
**Leaderboard**
This leaderboard is calculated with approximately 70% of the test data. The final results will 
be based on the other 30%, so the final standings may be different.

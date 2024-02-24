# Sudoku Digit Recognition

## Project Description
The project objective is to train a deep learning model that can recognize digits within a Sudoku puzzle and identify their positions within the grid. This repository contains the implementation for Task 1 of the project, which focuses on training a model to read the Sudoku puzzle.

## Tasks
1. **Task 1: Digit Recognition** - Train a model to recognize digits within a Sudoku puzzle and identify their positions within the grid.
2. **Task 2: Puzzle Solving** - Train a model to solve the Sudoku puzzle.

## Task 1: Digit Recognition
For Task 1, the goal is to predict the correct label for each digit in all the test images. In addition to the digit class, the model needs to submit each digit's cell position in the puzzle.

### Dataset
The dataset provided includes:
- Training Images: NumPy array of size 50000x252x252.
- Training Labels: NumPy array of size 50000x41x3, where the variables are arranged as x,y,value.
- Test Images: NumPy array of size 10000x252x252.

### Evaluation
The evaluation metric is based on predicting the digits filled within the puzzle and their positions. The submission file should contain values for all 81 cells in the puzzle, with the recognized digit for the filled cells and zero for empty cells, along with their x,y coordinates for all test images.

### Submission Format
The submission file should contain 81 values for each test image, with the format:
```
id,value
0_00,0
0_01,1
0_02,5
...
9999_88,9
```
In total, there shall be 810001 lines in the submission file including the header.

## Usage
1. Download or mount the dataset provided via the Google Drive link.
2. Preprocess the data if necessary.
3. Train the digit recognition model using the provided training images and labels.
4. Generate predictions for the test images.
5. Create a submission file following the specified format.
6. Submit the predictions for evaluation.

## Files Included
1. `train_digit_recognition.ipynb`: Jupyter notebook containing code for training the digit recognition model.
2. `test_digit_recognition.ipynb`: Jupyter notebook containing code for testing the digit recognition model and generating predictions.
3. `submission.csv`: Sample submission file.

## Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- pandas
- Matplotlib

## References
- [Data Set](https://drive.google.com/drive/folders/1iyQDn_kE_QIgGvtxMRcBVqf_Gcxe9Pua?usp=drive_link) - Dataset and competition details.

## Note
- Use  **Google colab** for  better  experience  of the  Repository.

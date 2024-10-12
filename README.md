# Sudoku Solver Using Neural Networks 🧩🤖

## Overview 🌟
This project aims to develop a Sudoku solver using neural networks. The project is divided into two main tasks, each comprising two methodologies:

1. **Task 1**: Train a model to recognize filled digits in Sudoku puzzles. 🔢
2. **Task 2**: Develop a model to complete Sudoku puzzles by filling in empty cells according to standard Sudoku rules. ✔️

This document outlines the data used, methodologies for each task, and results obtained from the models. 📊

## Data 📚
The project utilizes datasets that include both images of Sudoku puzzles and their corresponding labels.

### Task 1 Data 🖼️
- **Training Data**: A dataset containing **50,000 images** of partially solved Sudoku puzzles. Each image represents a grid with some digits filled in. 🏞️
- **Labels**: Each image is annotated with the positions of the 41 filled cells. 🏷️

### Task 2 Data 🖼️
- **Training Data**: A dataset of **50,000 pairs** of images, where each pair consists of a partially filled Sudoku image and its completely filled counterpart. 🔄
- **Test Data**: A set of **10,000 images** of partially filled Sudoku puzzles for evaluation. 🔍

Here's the updated version of the **Methodology** section for Task 2 in your Sudoku Solver project documentation. This incorporates your specified steps about using a pre-trained digit recognition model based on the MNIST dataset. I’ve added plenty of emojis for an engaging touch! 

---

## Methodology 🛠️

### Methodology of Task 1 🔢
#### Task 1: Recognizing Digits in Sudoku
1. **Data Preprocessing** 📥:
   - **Loading Data**: The dataset is loaded using NumPy to facilitate easy manipulation. 📊
   - **Image Resizing**: Each image is resized to a consistent format (e.g., 28x28 pixels) to isolate individual digits, ensuring uniformity for model training. 🖼️
   - **Normalization**: Pixel values are normalized (scaled to the range [0, 1]) to improve the training process's efficiency and convergence speed. ⚡
   - **Data Augmentation**: Techniques like rotation, translation, and scaling may be applied to increase dataset diversity and help the model generalize better. 🔄

2. **Model Architecture** 🏗️:
   - **Input Layer**: The model takes in flattened 28x28 pixel images (784 input nodes). 🖼️
   - **Hidden Layers**: Two hidden layers with varying numbers of neurons (e.g., 128 and 64) using ReLU (Rectified Linear Unit) activation to introduce non-linearity. 🧠
   - **Output Layer**: A softmax output layer with 10 neurons corresponding to digit classes (0-9). 🔟

3. **Training Process** 📈:
   - **Optimizer**: The Adam optimizer is used for its efficiency and adaptability. 🏎️
   - **Loss Function**: CrossEntropyLoss is employed, which is suitable for multi-class classification tasks. ⚖️
   - **Training Epochs**: The model is trained over a specified number of epochs (e.g., 20), with mini-batch gradient descent for effective learning. ⏳
   - **Validation**: A validation dataset (e.g., 20% of the training data) is used to monitor performance during training, enabling early stopping if overfitting occurs. 🚦

4. **Results and Evaluation** 📊:
   - **Performance Metrics**: Accuracy and loss are calculated on the test dataset to evaluate the model’s performance. 🏆
   - **Confusion Matrix**: A confusion matrix is generated to visualize the model's classification results, showing true positives, false positives, and false negatives for each digit. 📉
   - **Submission File**: A CSV file containing predicted digits and their positions is generated for evaluation. 🗂️

### Methodology of Task 2 🧩
#### Task 2: Completing Sudoku Puzzles
1. **Data Loading** 📥:
   - **Image Pairs**: The training images (partially filled grids) and labels (fully filled grids) are loaded into memory, ensuring both sets correspond correctly. 🖼️
   - **Normalization**: Similar to Task 1, pixel values are normalized to enhance model training performance. ⚡

2. **Pre-trained Digit Recognition Model** 🔍:
   - **Training on MNIST Dataset**: The digit recognition model is first trained on the MNIST dataset, which contains handwritten digits (0-9). This serves as the foundational step for recognizing digits in Sudoku puzzles. 📚
   - **Model Architecture**: A convolutional neural network (CNN) is employed to learn features from the MNIST dataset effectively. This model will later be fine-tuned for Sudoku digit recognition. 🧠
   - **Transfer Learning**: The trained MNIST model will be utilized to recognize digits' locations in the Sudoku puzzle images. The weights of the pre-trained model are frozen initially, allowing for faster inference. 🔄

3. **Digit Recognition in Sudoku** 🧩:
   - **Using the Pre-trained Model**: The pre-trained digit recognition model is applied to the Sudoku puzzle images to identify and locate filled digits within the grid. The model outputs the coordinates of the recognized digits. 🗺️
   - **Digit Extraction**: The locations of identified digits are extracted, creating a structured representation of the Sudoku puzzle's filled cells. 📊

4. **Model Architecture for Puzzle Completion** 🏗️:
   - **Input Layer**: The model takes in 252x252 pixel images of Sudoku grids. 🖼️
   - **Convolutional Layers**: Several convolutional layers are used to capture spatial hierarchies in the input data. Each convolutional layer is followed by a ReLU activation function and a max-pooling layer to reduce dimensionality. 📉
   - **Fully Connected Layers**: After several convolutional layers, the output is flattened and fed into fully connected layers to produce predictions for each digit in the Sudoku grid. 🔗

5. **Training the Solver Model** 📈:
   - **Training Process**: The puzzle completion model is trained using the filled Sudoku grid as labels. The model learns to predict the missing digits based on the current state of the grid. 📚
   - **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and the number of epochs to find the optimal configuration. ⚙️
   - **Early Stopping**: Implement early stopping to halt training if validation loss does not improve for a defined number of epochs. 🚦

6. **Making Predictions** 🔮:
   - **Inference**: The trained model is used to predict the digits for the unseen test dataset to complete the Sudoku puzzles. 🧙‍♂️
   - **Output Processing**: Predictions are compiled into a structured format, ensuring the predicted solutions align with the original puzzle format. 📜
   - **File Generation**: The results are saved in a CSV file named `solved_Submission.csv`, ready for evaluation. 📁

## Results 🎉
The models were trained and evaluated on the respective datasets. Key findings include:
- The digit recognition model achieved an accuracy of **X%** on the test dataset. 🏆
- The Sudoku completion model successfully filled in missing numbers with an accuracy of **Y%**. ✅

Visualizations, including loss and accuracy plots, provided insight into model performance during training. 📉📈

## Conclusion 📝
The project successfully demonstrates the use of neural networks for solving Sudoku puzzles. Both tasks provided valuable insights into image recognition and classification techniques. The models can be further improved by tuning hyperparameters and exploring more advanced architectures. 🚀

## Acknowledgments 🙏
- This project was developed as part of the Artificial Neural Networks and Cognitive Models course at THWS University, Germany . 🎓
- Special thanks to the course instructors and peers for their support and guidance throughout the project. ❤️

## License 📜
This project is licensed under the MIT License. See the LICENSE file for details. 📄

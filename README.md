# X-Ray Image Classification Using Deep Learning

## Project Overview

This project is focused on the classification of chest X-ray images to diagnose medical conditions such as pneumonia. It utilizes deep learning models built with TensorFlow and Keras, employing convolutional neural networks (CNNs) for feature extraction and classification.

## Features

- Image preprocessing and augmentation to enhance model training.
- Use of Convolutional Neural Networks for accurate image classification.
- Evaluation of model performance with metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Visualization of training progress, confusion matrix, and ROC curve.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6+
- Libraries: TensorFlow, Keras, NumPy, OpenCV, Scikit-Learn, Matplotlib, Seaborn

## Installation

To install the required libraries, run the following command:

```bash
pip install numpy opencv-python tensorflow keras scikit-learn matplotlib seaborn
```

## Usage

1. **Data Preparation**: Place your dataset in the `data/chest_xray/train` directory. The dataset should be divided into subfolders representing each class.

2. **Model Training**: Run the script to train the model. This will automatically split the data, preprocess it, and augment it before feeding it into the model.

3. **Model Evaluation**: After training, the model will evaluate the test set, outputting metrics such as accuracy and generating a confusion matrix and ROC curve.

## Running the Code

To execute the program, navigate to the directory containing the script and run:

```bash
python classification.ipynb
```

Ensure that your dataset is structured correctly, as the script will look for images within the specified `data_dir`.

## Contributing

Interested in contributing? Great! Please feel free to submit pull requests or open issues for any improvements or bug fixes.

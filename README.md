# Brain Tumor Segmentation using U-Net

This project implements an image segmentation technique for detecting brain tumors using U-Net architecture from scratch. The dataset used for this project is from Kaggle: [Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/code).

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Description
This project aims to detect brain tumors in MRI images by performing image segmentation using the U-Net model. The U-Net architecture is well-suited for medical image segmentation due to its ability to capture both local and global features efficiently. 

The model was built from scratch using Python and TensorFlow/Keras, and trained on the Kaggle Brain Tumor Segmentation dataset. The primary goal is to segment the brain tumor regions in MRI scans for better diagnosis and analysis.

## Dataset
The dataset used for this project is the Brain Tumor Segmentation Dataset, which can be found on Kaggle: [Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/code).

### Dataset Details:
- **Images**: MRI scans of the brain (typically in `.nii` or `.png` format)
- **Labels**: Corresponding tumor segmentation masks (binary masks indicating the tumor region)
- **Training/Testing Split**: The dataset is split into training and test sets.

## Model Architecture
The segmentation model used is a U-Net architecture built from scratch. It consists of the following key components:
- **Encoder**: A series of convolutional layers that extract features from the input image.
- **Bottleneck**: The central part of the U-Net where the most abstract features are learned.
- **Decoder**: A series of up-sampling layers that combine high-level features from the encoder with spatial information from earlier layers to accurately segment the image.

### U-Net Features:
- Skip connections that help retain spatial information.
- Symmetric architecture with an equal number of encoding and decoding layers.
- Output layer with a sigmoid activation to predict binary masks for tumor segmentation.

## Dependencies
The following libraries are required to run this project:

- Python 3.x
- TensorFlow (Keras)
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- Kaggle (for downloading the dataset)

You can install the required libraries by running:

```bash
pip install -r requirements.txt

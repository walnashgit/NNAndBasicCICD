# MNIST Classification with CI/CD Pipeline

[![ML Pipeline](https://github.com/walnashgit/NNAndBasicCICD/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/walnashgit/NNAndBasicCICD/actions/workflows/ml-pipeline.yml)

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model architecture uses batch normalization, dropout, and global average pooling to achieve >95% accuracy on the test set.

## Model Architecture
- Input Layer: Accepts 28x28 grayscale images
- 5 Convolutional layers with batch normalization and dropout
- Global Average Pooling
- Final Dense layer with 10 outputs (one for each digit)
- Total parameters: < 25,000

## Features
- Automated training and testing pipeline
- Model validation checks:
  - Parameter count verification (< 25,000 parameters)
  - Input shape validation (28x28)
  - Output shape validation (10 classes)
  - Accuracy threshold (> 95% on test set)
- Automatic model versioning with timestamp and accuracy
- CPU-only training support for both local and CI environments

## Requirements
- Python 3.8 or higher
- PyTorch (CPU version)
- Other dependencies listed in requirements.txt

## Local Setup

1. Clone the repository:

```
git clone https://github.com/walnashgit/NNAndBasicCICD.git
cd NNAndBasicCICD
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate 
On Windows: 
venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Train the model:

```
python train.py
```

5. Run tests:

```
python -m pytest test_model.py
```


## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:
1. Sets up a Python environment
2. Installs dependencies (CPU-only versions)
3. Trains the model
4. Runs validation tests
5. Archives the trained model as an artifact

The pipeline is triggered on every push to the repository.

## Model Training Details
- Dataset: MNIST (60,000 training images, 10,000 test images)
- Batch Size: 128
- Optimizer: Adam (lr=0.01)
- Loss Function: Negative Log Likelihood
- Training: Single epoch with progress bar showing loss and accuracy
- Model Saving: Automatic with timestamp and accuracy in filename

## Testing
The test suite (`test_model.py`) verifies:
- Model architecture compatibility with 28x28 input images
- Total parameter count (< 25,000)
- Model accuracy on test set (> 95%)
- Output shape (10 classes)

## Model Artifacts
Trained models are saved in the `models/` directory with the name mnist_model_YYYYMMDD_HHMMSS_accXX.X.pth
where:
- YYYYMMDD: Date
- HHMMSS: Time
- XX.X: Achieved accuracy
# Deep Learning Project

This project focuses on training and evaluating various deep learning models for image classification tasks, specifically targeting COVID-19 dataset. The models include CNN-based architectures and ResNet variants.

## Table of Contents
- [Deep Learning Project](#deep-learning-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)

## Installation
To install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Kostas-Xafis/DeepLearning
    cd DeepLearning
    ```

2. Create a virtual environment and activate it (if needed):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
[!IMPORTANT]
Before running the training script, make sure to load the dataset into the `dataset` directory. The dataset should be organized as follows:
Example: `dataset/COVID-19/Radiography_Dataset/COVID/COVID-1.png`

To train and evaluate a model, use the `training.py` script. Below is an example command to run the script:

```bash
python training.py

Choose a model to use: 
(1.) Cnn1
(2.) Cnn2
(3.) ResNet50
(4.) ResNet50 - Pretrained
(5.) BasicBlock based CNN
Enter a number: [_]

Use the full dataset or a smaller one (1:Full or 2:Small): [_]

----Estimated VRAM usage----
  Training data: 13.62 GB
  Validation data: 4.54 GB
  Training & Validation data: 18.16 GB

Would you like to load the full dataset into the GPU memory?
(1.) None
(2.) Training data
(3.) Validation data
(4.) Training & Validation data
(5.) Resize the images (1-100%) - Current size: 299x299
Εnter a number: [_]

```

## Command Line Arguments
The following arguments can be passed to the `training.py` script:

- `--model`: Specifies the model to use. Options are `cnn1`, `cnn2`, `resnet50`, `resnet50-pretrained`, `cnn-basicblock`.
- `--epochs`: Number of epochs to train the model. Default is 10.
- `--batch_size`: Batch size for training. Default is 64.
- `--lr`: Learning rate for the optimizer. Default is 0.001.
- `--verbose`: If set, enables verbose logging.
- `--log`: If set, logs the output to a file.
- `--save_fig`: If set, saves the confusion matrix plot.

Example usage with all arguments:

```bash
python training.py --model resnet50 --epochs 10 --verbose --log --save_fig
```

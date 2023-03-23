# Udacity AI Python Nanodegree Final Project: Image Classifier

This project is part of the Udacity AI Python Nanodegree program. The goal of this project is to build an image classifier using deep learning techniques in Python. The classifier is trained on a dataset of flower images and is able to predict the class of a new image with high accuracy.

## Getting Started

To run the classifier, you will need to download the following files and directories:

- `train.py`: Python script for training the classifier on the flower image dataset.
- `predict.py`: Python script for using the trained classifier to predict the class of a new image.
- `cat_to_name.json`: JSON file containing the mapping of flower category labels to names.
- `flowers/`: Directory containing the training, validation, and test image datasets.

## Training the Classifier

To train the classifier, run the following command in the terminal:

`python train.py flowers/ --save_dir checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20`


This command trains the classifier on the flower image dataset using the VGG16 architecture with a learning rate of 0.001, 512 hidden units in the fully connected layer, and for 20 epochs. The trained model is saved in a checkpoint file called `checkpoint.pth`.

## Predicting Image Classes

To use the trained classifier to predict the class of a new image, run the following command in the terminal:

`python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json`


This command predicts the top 5 most likely classes for the image `flowers/test/1/image_06743.jpg` using the trained model stored in `checkpoint.pth` and maps the category labels to names using the `cat_to_name.json` file.

## Additional Information

For more information on the project, please refer to the Jupyter Notebook file `Image Classifier Project.ipynb`.

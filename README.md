# Mouse
Controlling Computer Mouse Using Hand Recognition


## Classification part
One of the essential parts of this project is how to classify considered classes and remove other classes. For the classification part which is based on a close dataset with no others, a model is trained by [6270HandImages](https://github.com/Youlenda/6720HandImages) dataset. The dataset has 5120 training and 1600 validation samples with different distribution and the validation set is more likely to real-time data. In addition, the 50% of validation set is allocated to test data. Therefore, the number of images in training, validation and test set are 5120, 800 and 800 images.

The dataset is classified by EfficientNet-B0 and some fully connected layers in order to separate considered hand poses. Python code for classification is in [dataset classification](https://github.com/Youlenda/Mouse/blob/master/dataset_classification.ipynb). Accuracy of the test set is 100% because the distribution of 
the validation set is simpler than the training set.

## Similarity part
One of classification problems is samples from unwanted classes that appear in real-time. Because of softmax activation function, each sample (from considered classes or not) will map to one of the last layer neurons. In this tutorial, we present a similarity neural network to compare new samples with dataset in order to remove unwanted classes.

# Mouse
Controlling Computer Mouse Using Hand Recognition


<p align="left">
  <img width="960" height="540" src="https://github.com/Youlenda/Mouse/blob/master/videos/on%2C%20tracking%20mode.gif">
</p>


## Classification part
One of the essential parts of this project is how to classify considered classes and remove other classes. For the classification part which is based on a close dataset with no others, a model is trained by [6270HandImages](https://github.com/Youlenda/6720HandImages) dataset. The dataset has 5120 training and 1600 validation samples with different distribution and the validation set is more likely to real-time data. In addition, the 50% of validation set is allocated to test data. Therefore, the number of images in training, validation and test set are 5120, 800 and 800 images.

The dataset is classified by [EfficientNet-B0](https://arxiv.org/abs/1905.11946) and some fully-connected layers in order to separate considered hand poses. Python code for classification is in [dataset classification](https://github.com/Youlenda/Mouse/blob/master/classification/dataset_classification.ipynb). Accuracy of the test set is 100% because the distribution of the validation set is simpler than the training set.

## Similarity part
One of classification problems is samples from unwanted classes that appear in real-time. Because of softmax activation function, each sample (from considered classes or not) will map to one of the last layer neurons. In this tutorial, we present a similarity neural network to compare new samples with dataset in order to remove unwanted classes.

The last layer of [EfficientNet-B0](https://arxiv.org/abs/1905.11946) has 1280 neurons and after training the model in the classification part, we remove fully-connected layers and provide a new model as a feature extractor; Samples of the validation set are imported to the feature extractor model. Then the average of encoded samples of each class calculate to create 4 reference vectors. In order to have 4 thresholds (for each class), we compare each encoded sample with its reference vector. Python code for classification is in [similarity neural network](https://github.com/Youlenda/Mouse/blob/master/classification/similarity_nn.ipynb)

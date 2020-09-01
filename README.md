# Mouse
Controlling Computer Mouse Using Hand Recognition


## Classification part
One of essential parts of this project is how to classify consideres classes and remove other classes. For classification part which is based on a close dataset with no others, model is trained by [6270HandImages](https://github.com/Youlenda/6720HandImages) dataset. The dataset has 5120 training and 1600 validation samples with different distribution and validation set is more likely to real-time data. In addition, 50% of validation set is allocated to test data. Therefore, number of images in training, validation and test set are 5120, 800 and 800 images.

The dataset is classified by EfficientNet-B0 and some fully connected layers in order to separate considered hand poses. Python code for classification is in [dataset classification](). Accuracy of test set is %.

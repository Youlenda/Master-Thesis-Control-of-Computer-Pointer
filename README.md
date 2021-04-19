# Master-Thesis-Control-of-Computer-Pointer

Control of Computer Pointer Using Hand Movement Detection in Motion Pictures.

**Please zoom in the screen for more details.**


<p align="left">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/on%2C%20tracking%20mode.gif">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20click.gif">
</p>


<p align="left">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20right-click.gif">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20off.gif">
</p>



**Abstract**

Humans have dreamed about picking things up from a long time ago by pointing to them with a single hand gesture. Up to now, no one could move objects without having direct touch, though artificial intelligence allows individuals to control the intelligent system by using hand gestures in front of a camera and without a single contact. 

In the proposed thesis, a user interface is designed to establish an interaction between humans and computers so the pointer can be controlled using hand gestures. These gestures are being captured and processed through a webcam, as well as applying image processing methods on them. Afterward, the hand location is detected by neural network algorithms. 

A hand dataset is collected with 6720 image samples, including four classes: fists, palms, pointing to the left, and pointing to the right. The images are captured from 15 individuals in simple backgrounds and different light conditions. The collected dataset trains a convolutional neural network based on EfficientNet-B0 and fully-connected layers. The trained network is saved for two purposes: First, to classify the output frames of a hand detector and predict a label for each frame; Second, to compare the output frames with images of the dataset.

Eventually, the predicted label converts to a command for controlling the computer pointer. The defined commands are turning the application on or off, moving the pointer, left and right-clicking. The algorithm reaches  92.6 per cent accuracy and is appropriate for use in different simple backgrounds.

**keywords**: Hand Gesture Recognition, Dataset, Convolutional Neural Network, Classification, Computer Pointer, and Object Detection


## Classification part
One of the essential parts of this project is how to classify considered classes and remove other classes. For the classification part which is based on a close dataset with no others, a model is trained by [6270HandImages](https://github.com/Youlenda/6720HandImages) dataset. The dataset has 5120 training and 1600 validation samples with different distribution and the validation set is more likely to real-time data. In addition, the 50% of validation set is allocated to test data. Therefore, the number of images in training, validation and test set are 5120, 800 and 800 images.

The dataset is classified by [EfficientNet-B0](https://arxiv.org/abs/1905.11946) and some fully-connected layers in order to separate considered hand poses. Python code for classification is in [dataset classification](https://github.com/Youlenda/Mouse/blob/master/classification/dataset_classification.ipynb). Accuracy of the test set is 99% because the distribution of the validation set is simpler than the training set.

## Similarity part
One of classification problems is samples from unwanted classes that appear in real-time. Because of softmax activation function, each sample (from considered classes or not) will map to one of the last layer neurons. In this tutorial, we present a similarity neural network to compare new samples with dataset in order to remove unwanted classes.

The last layer of [EfficientNet-B0](https://arxiv.org/abs/1905.11946) has 1280 neurons and after training the model in the classification part, we remove fully-connected layers and provide a new model as a feature extractor; Samples of the validation set are imported to the feature extractor model. Then the average of encoded samples of each class calculate to create 4 reference vectors. In order to have 4 thresholds (for each class), we compare each encoded sample with its reference vector. Python code for classification is in [similarity neural network](https://github.com/Youlenda/Mouse/blob/master/classification/similarity_nn.ipynb)

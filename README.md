# Master-Thesis-Control-of-Computer-Pointer

Control of Computer Pointer Using Hand Movement Detection in Motion Pictures.

<p align="center">
     Please zoom in the screen for more details.
</p>



<p align="left">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/on%2C%20tracking%20mode.gif">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20click.gif">
</p>


<p align="left">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20right-click.gif">
  <img width="384" height="216" src="https://github.com/Youlenda/Mouse/blob/master/videos/tracking%20mode%2C%20off.gif">
</p>



## Abstract

Humans have dreamed about picking things up from a long time ago by pointing to them with a single hand gesture. Up to now, no one could move objects without having direct touch, though artificial intelligence allows individuals to control the intelligent system by using hand gestures in front of a camera and without a single contact. 

In the proposed thesis, a user interface is designed to establish an interaction between humans and computers so the pointer can be controlled using hand gestures. These gestures are being captured and processed through a webcam, as well as applying image processing methods on them. Afterward, the hand location is detected by neural network algorithms. 

A hand dataset is collected with 6720 image samples, including four classes: fists, palms, pointing to the left, and pointing to the right. The images are captured from 15 individuals in simple backgrounds and different light conditions. The collected dataset trains a convolutional neural network based on EfficientNet-B0 and fully-connected layers. The trained network is saved for two purposes: First, to classify the output frames of a hand detector and predict a label for each frame; Second, to compare the output frames with images of the dataset.

Eventually, the predicted label converts to a command for controlling the computer pointer. The defined commands are turning the application on or off, moving the pointer, left and right-clicking. The algorithm reaches  92.6 per cent accuracy and is appropriate for use in different simple backgrounds.

**keywords**: Hand Gesture Recognition, Dataset, Convolutional Neural Network, Classification, Computer Pointer, and Object Detection


## Classification Part
Like any other image classification task, this project should deal with how to classify a sample to one of the defined classes or ignore that if it belongs to an undefined one. For the classification part, which is based on a close dataset as opposed to having all the possible classes, a model is trained by [6720HandDatase](https://github.com/Youlenda/6720HandDataset), having a limited number of samples as well as classes. The dataset has 5120 training and 1600 validation samples (800 for validation, 800 for test). Although it is common to split training and validation sets randomly, in this project, they have different distribution to boost the generalization power since validation samples are more likely to in real-time frames. The dataset is classified by [EfficientNet-B0](https://arxiv.org/abs/1905.11946) and some fully-connected layers in order to distinguish between defined hand gestures. The model reaches 99 per cent accuracy due to the simpler distribution of the validation set rather than the training set. Python code for classification is in [here](https://github.com/Youlenda/Mouse/blob/master/classification/dataset_classification.ipynb).

## Similarity Part
One of the main problems of image classification is that there will be frames from undefined (unwanted) classes in real-time, which algorithms must learn to ignore, especially for hand applications in which a hand can make a plethora of gestures.

Because of the SoftMax activation function, any captured frame (from the dataset classes or not) would be mapped to one neuron of the last layer. Thus, it would be essential to remove undesirable frames before going through the classification part. In fact, a similarity neural network is designed to compare new samples with the dataset in order to remove unwanted classes before reaching the classification part.

The architecture for classification is considered [EfficientNet-B0](https://arxiv.org/abs/1905.11946) with fully-connected layers. When the model in the classification part was trained, the model was frozen and fully-connected layers were removed to provide a new model as a feature extractor. The validation set was then imported to the feature extractor model. When the samples were encoded, the average of them for each class was calculated to create four reference vectors. For having one threshold for each class, each encoded sample was compared with its reference vector.  Python code for the similarity part is in [similarity neural network](https://github.com/Youlenda/Mouse/blob/master/classification/similarity_nn.ipynb)

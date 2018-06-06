This repository contains my solution to the semantic segmentation challenge organized by Lyft and Udacity. This 1-month competition was opened to students of Udacity's Self-Driving Car Nanodegree and the participants were asked to label each pixel in an image as a car, the road or something else. The dataset of the competition was gathered from the [CARLA simulator](https://github.com/carla-simulator/carla/blob/master/README.md) and the participants were allowed to use the simulator to collect more data.

Here is the result of my model. I'm proud to rank 3rd out of 155 entries on the final leaderboard without prior semantic segmentation knowledge.

![alt text][annotated]

I used a U-Net[1][2] like network which has an encoder based on pre-trained ResNet34. The decoder consists of several decoder blocks, each of which includes transposed convolution, relu activation and batch normalization. The output of each decoder block is concatenated with corresponding layer from the encoder before being passed to the next decoder block. A transposed convolution is added in the end of the network to upsample the output to the input shape.

I combined softmax cross entropy loss and intersection over union to one loss function as described in [2], and used Adam optimizer and trained the network with a learning rate of 0.001. The submitted model was trained with a training set consisting of 24517 cropped images for 10 epochs and had a 99.5% accuracy on a validation set of 2710 images. Both the training set and validation set include images from the provided dataset and the frames that I captured with the CARLA simulator using different settings.

[1] O. Ronneberger, P. Fischer and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597), 2015.\
[2] V. Iglovikov and A. Shvets, TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation, [arXiv: 1801.05746](https://arxiv.org/abs/1801.05746), 2018.

[//]: # (Image References)

[annotated]: ./annotated300x400.gif
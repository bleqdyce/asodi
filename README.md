# A Simple Object Detection Introduction

I record what I have read and survey here, and try to stay organized and friendly as much as I can.

## Object Detection

Generally speaking, It's a task about to find a way to locate certain object on the image and tell what the object is automatically.
e.g., Use Neural Network to find out **Where's Wally / Wildo** like [this](https://towardsdatascience.com/how-to-find-wally-neural-network-eddbb20b0b90) and [this](https://github.com/alseambusher/deepwaldo).

![Wally](./wally_detected.jpg)

### Terminology

Usually, You will see Object Detection along with several keywords like Machine Learning, Computer Vision, Image Classification, Neural Network, CNN ... and so on. To clarify this, I want to explain the relationships between these words here .

But keep in mind that my explanation may not be comprehensive enough because every words I'm going to mention could be a topic that would spend you months to study.

- **Machine Learning** is a study about how to design a clever program that can automatically find out how to solve various tasks without human hand-holding.
- **Computer Vision** is a sub domain of Machine Learning. Its goal is pretty much same with Machine Learning but only focus on the tasks those process image or video.
  - **Image Classification** is a topic / task of Computer Vision whose goal is to find a way to tell what the object on the image is automatically.
  - There are still other tasks like **Image Segmentation**, **Image Captioning** ...
- **Neural Network**
  - **Convolutional Neural Network (CNN)** is one kind of Neural Network structure which is specially design to solve Computer Vision tasks mostly. The core idea is all about using **convolutional layer** to simulate how human brian process what people see.
- **Dataset**
- **TensorFlow**
- **Classifier**
- **Training**

<!-- 
Object Detection is a very popular research domain which is talking about how to find a better way to detect objects on images automatically. Most recent approaches rely on **Machine learning** (, or **Neural Networks** if more specific).
### Machine Learning?
Machine Learning is a very big research domain which is more popular than **Object Detection**. The idea is about to design a clever way to let the machine or program could find a **magical formula** which is able to solve certain task without human hand-holding. 
### Magic Formula?
In Machine Learning, We call the **Magic Formula** as **Model**. Keep in mind, almost every model was designed to solve a very specific problem and that's the main reason why it is a Magic Formula instead of just Magic
### Neural Network?
-->

### Overview

I Split this section into 2 parts, One is **Before Neural Network** and another is **Neural Network**. In First part, I will briefly introduce some method which was either common or once state-of-the-art at their time. Then I will introduce **Neural Network** which is actually the reason make me to start writing this in next part.

#### Before Neural Network

As far as I know, Most traditional method around 2006 - 2010 could fit into the **three steps flow**. See the figure below.

![three steps flow](./3_steps.png)

To put it simply, this is what actually each steps do:

1. **Region Proposal** - Find those region may potentially contains an object. The most common and easiest method is **sliding window**. Just like the figure shown below, a fixed-size window would slide over entire image and the bounded area would be treated as the proposed regions.

![sliding window](./sliding_window_example.gif)
2. **Feature Extraction** - Transform the proposed regions of image into a different kind of representation which is usually defined by human. For example, Histogram of Gradient (HoG) will calculate the gradient between pixels and take is as a representation of image. It was once very popular because its promising performance on object detection.

![HOG](hog.png)
3. **Classification** - Determine if the proposed area does contain an object according to transformed representation. In this step, we would need a classifier which is trained with transformed representation to achieve this task. I'm not going to explain how to train a classifier here. Therefore, if you haven't studied Machine Learning but you are interested in it, it is much better that taking a Machine Learning course on MOOC site e.g., Coursera or Udacity.

Here is some keywords of traditional method, I only list few because I know very less about them.

1. Region Proposal- Sliding Window
2. Feature Extraction - Haar, HOG, LBP, ACF
3. Classification - SVM, AdaBoost, DPM

#### Neural Network

Generally, NN-based method could be  classified into 2 classes. One is often called as **Two-Stage Detectors** due to the way it approach the task. They first find regions that are potential to be a object over image and then try to tell what kind of object is it. Another is **One-Stage Detectors** which attempt to solve two problems together.

Some also says that the major difference between the two kind is how they approach the problem. Two-Stage Detector try to take Object Detection as a classification problem and One-Stage Detector treat it as a regression problem.

Here is the paper list of both kinds detectors. I only list those I've heard of, If I miss something important, remind me please.

- Two-Stage Detectors
  - [R-CNN](https://arxiv.org/abs/1311.2524)
  - [SPP-Net](https://arxiv.org/abs/1406.4729)
  - [Fast R-CNN](https://arxiv.org/abs/1504.08083)
  - [Faster R-CNN](https://arxiv.org/abs/1506.01497)
  - [R-FCN](https://arxiv.org/abs/1605.06409)
  - [Feature Pyramid Networks (FPN)](https://arxiv.org/abs/1612.03144)
  - [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
  - [Mask R-CNN](https://arxiv.org/abs/1703.06870)
  - [RetinaNet](https://arxiv.org/abs/1708.02002)
  - [Light-Head R-CNN](https://arxiv.org/abs/1711.07264)
- One-Stage Detectors
  - [OverFeat](https://arxiv.org/abs/1312.6229)
  - [YOLO](https://arxiv.org/abs/1506.02640)
  - [SSD](https://arxiv.org/abs/1512.02325)

## Benchmarks / Datasets

- [COCO](http://cocodataset.org)
- Pascal VOC [2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) / [2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
  <!--- aeroplane, bicycle, bird, boat, bottle, bus, car, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor-->
- [ImageNet Det](http://www.image-net.org/challenges/LSVRC/)

![comparison1](./dataset_comparison_1.jpg)
![comparison2](./dataset_comparison_2.jpg)
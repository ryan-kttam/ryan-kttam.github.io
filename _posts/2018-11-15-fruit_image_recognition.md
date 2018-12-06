---
title: "Fruit Image Recognition"
date: 2018-11-15
header:
  image: "/images/fruit2.jpg"
excerpt: "Convolutional Neural Network, Machine Learning, Data Science"
---

## Introduction

In this post, I am going to use deep learning techniques to build a model that is able to identify what fruit is showing in an image. The purpose of this model is to help people get to know more about fruit's nutritional values, especially when they see fruits that they do not eat often. Building a nutritional database is definitely an crucial part of this project, but first I will need to build a model that is able to identify what fruit is showing in an image.

### Why Convolutional Neural Network?

<img src="/images/cnn.jpg" alt="example">

Human identify objects by analyzing their color, shape, and texture. Convolutional neural network is capable of completing such task, because it is able to preserve spatial relationship between pixels, which ultimately isolating color, shape, and texture as independent factors. it then can predict what fruit is in the image based on the factors identified by CNN.


## Data Preprocessing

I gathered 22 types of fruit images on Google, each type has its own folder and has approximately 100 images. Given this is a relatively small dataset, I decided to implement **data augmentation** in order to increase the amount of examples to train the model. There are four forms for each image: its original form, a flipped version, rotated 90 degrees, and rotated 270 degrees. I also resized all images to *100x100* pixels in order to increase the training speed performance, and divide all pixels by 255 in order to limit their ranges to 0 and 1.

```python
def process_image(img_path, img_list, label_list, fruit_name, dim):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (dim, dim))  # resize to dim*dim (dim=100 in my case)
    img_list.append((image / 255.).tolist())
    img_list.append((np.flipud(image)/ 255.).tolist()) # adding a flipped form
    img_list.append((rot(image, 90)  / 255.).tolist()) # adding a rotated image (90 degrees)
    img_list.append((rot(image, 270) / 255.).tolist()) # adding a rotated image (270 degrees)
    label_list += [fruit_name]*4 # add those fruit labels
```

## Model Tuning

I will be implementing a pre-trained model, **VGG16**, in this project. VGG16 is a CNN architecture that is considered to be an excellent vision model, and it generally performs well on object classifications. Prior to implementing VGG16, I have already tried out different models, such as artificial neural network and self-built CNN. It turns out that VGG16 performed the best among those three models.

I am going to fine-tune the model by adjusting the following:
* the number of trainable VGG16 layers (None vs. the last 5 layers)
* learning rates (0.01 vs. 0.001)
* RMSprop vs. SGD optimizer

At the end of the VGG16 model, I will be adding a fully connected layer (ReLU) as well as two dropout layers before and after the fully connected layer. I then close the model by adding a softmax layer with 22 nodes, which equal the number of fruit types. Adding dropout layers can reduce the model to overfit by preventing co-adaptation on data. In my case, the dropout rate is 20%, meaning the model will be randomly blocking 20% of the nodes before and after the fully connected layer when training the model. The image below shows the beginning and the last layers of the model.
<img src="/images/cnn_model.jpg" alt="mountain">

result:::
# None, 0.001, RMS: 72.56%
# None, 0.01, RMS: 3.72%
# None, 0.0001, RMS: 72.35%
# last 5, 0.001, RMS: 6.88%
# last 5, 0.01, RMS: 3.72%
# last 5, 0.0001, RMS: 81.73%
# last 5, 0.00001, RMS: 78.72%
# None, 0.001, SGD: 60.74%
# None, 0.01, SGD: 70.27%
# None, 0.0001, SGD: 33.46%
# last 5, 0.001, RMS: 76.65%
# last 5, 0.01, RMS: 79.51%
# last 5, 0.0001, RMS: 56.59%

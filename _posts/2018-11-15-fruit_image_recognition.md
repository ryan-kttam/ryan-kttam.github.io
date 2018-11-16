---
title: "Fruit Image Recognition"
date: 2018-11-15
header:
  image: "/images/fruit2.jpg"
excerpt: "Convolutional Neural Network, Machine Learning, Data Science"
---

## Introduction

The detailed paper is available at [here](/supporting_doc/Fruit_Image_Recognition_paper.pdf)

There are many fruits available in the market today. Consumers, however, may not know every fruit and their nutritional value. They can definitely search for nutrition value for any fruit. However, for people who do not use internet often (i.e. elder), they might have a hard time looking for the fruit's nutrition value.
Therefore, the goal of this post is to develop machine learning model, specifically a convolutional neural network, or CNN, to recognize what fruit is showing in an image.

### Why Convolutional Neural Network?
Human identify objects by analyzing their color, shape, and texture. Convolutional neural network is capable of completing such task, because it is able to preserve spatial relationship between pixels, which ultimately isolating color, shape, and texture as independent factors. it then can predict what fruit is in the image based on the factors identified by CNN.

## Analysis

There are 22 types of fruits, each with approximately 100 images from Google.

There are some images that can be confusing to the model:
<img src="/images/pineapple1.jpg" alt="multiple"> <img src="/images/pineapple2.jpg" alt="single">
For example, some images contain multiple pineapples (left), while some only contain one pineapple.

I will be implementing a pre-trained model, VGG16, in this project. VGG16 is a CNN architecture that is considered to be an excellent vision model, and it generally performs well on object classifications.

The last five layer ...


# H1 Heading

## H2 Heading

### H3 Heading

some basic text.

example with *italics*

example with **bold** text

example with [link](https://github.com/ryan-kttam)

example with bulleted list:
* first
+ two
- three

example with numbered list:
1. first
2. two

python code block:
```python
    import numpy as np
    def test_function(x):
      z = np.sum(x)
      return z
```

python code block:
```r
library(tidyverse)
df = read.csv('xxx.csv')
head(df)
```

inline code: `x+y`

an image:
<img src="/images/mm2.jpg" alt="mountain">

math equation:

$$ z=x+y$$

---
title: "Voice Recognition"
date: 2017-11-26
header:
  image: "/images/voice.jpg"
excerpt: "Support Vector Machine, Model Tuning, Machine Learning, Data Science"
---
# Please visit [here](https://github.com/ryan-kttam/Voice-Recognition-Project) for more information.

# Background

Suppose an advertising team on Facebook is interested in identifying a person’s gender from an audio clip in order to increase the effectiveness in advertisements, and the engineer teams have preprocessed and transformed the data into various formats. My task is to implement an algorithm that can classify whether the clip contains a male voice or a female voice.
The dataset is available at [Kaggle](https://www.kaggle.com/jeganathan/voice-recognition), and the python file is available at my [Github]( https://github.com/ryan-kttam/) page.

# Data Exploration

Checking whether I read the data correctly and whether there are any NA values.
``` python
  data.head(2)
  data.isnull().sum() # alternative: data[data.isnull().any(axis=1)]
  data['label'].value_counts() # there are 1584 male voices and 1584 female voices.
```

The dataset is imported correctly and there are no NA values. The dataset has a total of 3168 rows and 21 columns, with male clips and female clips equally distributed. To visualize how genders differ from each other, I generated several graphs using features like centroid and mean frequency. The more distinct they are, the easier it is to train a machine learning algorithm.

#####graphsssss
<img src="/images/voice_recognition/Figure_1.jpg" alt="Figure 1">
<img src="/images/voice_recognition/Figure_2.jpg" alt="Figure 2">

# Data Preprocessing
Standardizing data allows machine learning models to treat each feature equally. Data Standardization is always a good practice as it neutralizes variables that have large values. It can also minimize the risk of one feature from being too dominant compared to others. There are several methods to achieve this goal, such as transforming feature’s range to 0 and 1 or standardizing features by removing the mean and scaling to unit variance. In this project, I will standardize the data by applying StandardScaler to each column: each attribute will have mean of 0 and standard deviation of 1.
In addition, I also use train_test_split from sklearn in order to split the data into a training set and a test set, with 90% of the data as the training set, and the remaining 10% as the test set.
``` python
```

# Model Training & Tuning
After splitting the data into training and test set, I used support vector machine algorithm in order to predict Male/Female voices. In addition, I used 'accuracy_score' from sklearn.metrics as a metric to evaluate the performance. Accuracy Score compares the predictions with the actual result: it returns the number of correct predictions over the number of predictions. The higher the accuracy is (highest being as 1, lowest as 0), the better the performance. Accuracy is a reasonable metric due to the balance of each class.
``` python
```
I will be using k-fold cross-validation in order to prevent the data from overfitting. For model tuning, I will adjust C and Gamma in order to find out the best parameters for this data; specifically, I will generate a graph for 'C' and see how large ‘C’ is for the best accuracy.
``` python
```
When using the code above, we can see that the model performs best when C is in between 5 and 15. To find out the best estimate for C, I chose to zoom in the graph and locate the exact value of C that give me the maximum of accuracy.
``` python
```

#### grpahssss

It turns out that the model performs the best when C is 10. Now repeat the same steps above, but with Gamma as the parameter. Here I simplify the method a bit by adjusting gamma as (0.0001, 0.001, .01, .1, 1, 10, 100). It turns out the best Gamma is 0.15.
As a result, the best hyperparameters for SVM for this data is C = 10 and gamma = 0.15.

# Finalize the model

Before I make any prediction on the test set, I chose to run a k-fold validation in order to make sure the model is not overfitting. It turns out that the average accuracy for a 10-fold validation is 98.17%.
``` python
```
It is finally the time to plug the model in to the test set.
The SVM model performed exceptionally well, with 99.05% accuracy on our test set! While this is a very good model, there are some potential improvements that I could have try when tuning the models, such as trying kernel = 'linear' or 'poly'. In our case, we did not adjust the kernel at all, only using 'rbf', but 99% is still a decent accuracy for my test set.
Not only can the model be used in advertising team to increase the effectiveness in finding the right target to sell their products, but it can also be used in customer service team. For example, they can predict whether the voice of a complaint is male or female and construct an analysis on gender regarding whether one gender tends to have more complains than the other. Having this model is handy as it save the company tons of resources and time.

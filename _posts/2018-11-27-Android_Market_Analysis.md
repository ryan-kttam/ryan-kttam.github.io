---
title: "Android Market Analysis"
date: 2018-11-27
header:
  image: "/images/android.jpg"
excerpt: "Data Visualization, Data Analysis, Data Science"
---

# Background
How much do you know about the mobile app marketing? What category do you think has the most installs in the market? is it social media? or games? In this post, I am going to explore and learn more about the mobile app market, specifically, the Android Market.
I will be using a dataset on [Kaggle](https://www.kaggle.com/lava18/google-play-store-apps), which includes more than 10,000 Play Store apps. The dataset has size, ratings, category, and 9 more features for us to explore. Let's see what we can get out of from the dataset. The Python code will be available [here] and I also uploaded the dataset on my [Github Page].

*Before you installed all required packages before running the code.*

# Data Preprocessing and Exploration

To begin with, I need to make sure if I am reading the data correctly. I would also like to view how the data look like in order to have an idea about what we could do with it.
```python
  # reading the data
  data = pd.read_csv('C:/.../googleplaystore.csv')
  # print out the first two rows and make sure we imported correctly
  data.head(2)
  # check how many rows in the data
  print (len(data))
```
<img src="/images/android_analysis/image1.jpg" alt="read the data">

The data looks good! Checking for NAs next...
```python
  # check if there is any missing or null values
  data.isnull().sum() # out of 10841 rows, there are 1474 missing ratings.
  sum(data.Size == 'Varies with device') # there are 1695 rows does not have a size due to the app size varies with device.
```

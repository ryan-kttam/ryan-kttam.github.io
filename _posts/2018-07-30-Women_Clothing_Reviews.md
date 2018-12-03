---
title: "Women Clothing Reviews"
date: 2018-04-11
header:
  image: "/images/women_clothes.jpg"
excerpt: "Natural Language Processing, NLP, Text mining, Data Science"
---

# Please visit [here](https://github.com/ryan-kttam/women_clothing) for more information.

# Background

Product reviews are very important for a company, as most consumers read reviews online before purchasing a product, they trust it as much as personal recommendations, and most reviews are expressive and honest. Product reviews also give companies ideas about how they can improve their products to increase customer’s satisfaction. In this post, I am going to predict how satisfied a consumer is after purchasing a product, based on their product reviews, specifically, the text reviews.

# Data exploration

Lets read the data and see what it looks like.
``` python
data = pd.read_csv('C:/Github/women_clothing/Womens_Clothing_Reviews.csv', header=0)
data = data.rename(columns={list(data)[0]: 'count'})
data.head(2)
print ('Column Name: '+ str(list(data)))
```
<img src="/images/women_clothing/image1.jpg" alt="image 1">
<img src="/images/women_clothing/image2.jpg" alt="image 2">

The data is loaded successfully. As we can see in the dataset, there are columns such as "Recommended IND" and "Positive Feedback Count" in the dataset. However, I am going to ignore them as this project focuses on text analysis. That being said, let’s explore how the text length has an influence on ratings.

# Data Exploration

<img src="/images/women_clothing/Figure_1.jpg" alt="Figure 1">

The trend of this boxplot looks like a mini pyramid, as the average text length increases and reaches it’s top at rating 3, then descends at the same rate and finally reaches rating 5. It means that purchasers who rated their products as 3 may have given more details in the reviews compared to the ones who rated 1 or 5.

Word cloud is a method that shows the most frequent words in a text. We often can find interesting insight as we can see the reasons why consumers rank their items in a certain ratings.

<img src="/images/women_clothing/Figure_2.jpg" alt="Figure 2">
<img src="/images/women_clothing/Figure_3.jpg" alt="Figure 3">
<img src="/images/women_clothing/Figure_4.jpg" alt="Figure 4">
<img src="/images/women_clothing/Figure_5.jpg" alt="Figure 5">
<img src="/images/women_clothing/Figure_6.jpg" alt="Figure 6">

Some highlighted words for Rating 1 are, look, fit, fabric, and color. As expected, there are negative words such as disappointed, cheap, and unflattering. Surprisingly, positive words can be seen in this cloud like cute and beautiful.
Starting from rating 3, we no longer see any negative words in the cloud. While look, fit, fabric, and color still dominate the cloud in rating 3, we are seeing more positive words such as beautiful, pretty, and love.
The number of positive words once again grow as rating reaches rating 5. In fact, love and perfect take over and finally dominate the cloud.

# Sentiment Analysis

Those clouds helped us visualize what consumers generally think when writing reviews. We can further translate these messages into numeric values by implementing sentiment analysis. I am using SentimentIntensityAnalyzer from NLTK in order to calculate the sentiment scores for each sentence. I am only using three of the four values this function returned: neg, neu, and pos, which are negative, neutral, and positive, respectively. The sum of these three values equals to one. For example, “This dress is absolutely perfect” will return (0.0, 0.5, 0.5), which means .5 in both neutral and positive.

## graph

This bar chart portrays the percentage of negative emotion decreases as rating increases, while the percentage of positive emotion increases as rating increases. Neutral emotion is the major part of the reviews and it remains stable throughout all ratings.

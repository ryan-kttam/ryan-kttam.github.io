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
 <img src="/images/android_analysis/image2.jpg" alt="missing data">

Take away: There are 1474 missing data on ratings, and it is unknown why so many apps have unknown ratings.

Next lets check whether the 'size' are using the same unit.
```python
  size_unit = [unit[-1] for unit in data.Size]
  Counter(size_unit) # there are 1 app ending with '+', 8829 apps ending with MB and 316 apps ending with kb. (1695 apps varies with device)
```
There is one app ending with '+', 8829 apps ending with MB and 316 apps ending with KB (also 1695 apps are "varies with device"). Something seems odd with the app size unit as '+', I decided to take a look:
<img src="/images/android_analysis/image3.jpg" alt="data error">
If we look closely, we can see that all columns for that particular seems to shift left by one column. For simplicity, I decided to remove this app from the dataset.
```python
  data = data.loc[[i!='+' for i in size_unit] ,]
```

Next step is to standardize all app size and make sure they are using the same unit. In this case, I am changing all app size to MB. In addition, I am removing all the comma and the plus sign in 'Install', then converting them to numeric values.
```python
  size_unit = [unit[-1] for unit in data.Size]
  kb_loc = [i=='k' for i in size_unit] # find all indexes of KB
  mb_loc = [i=='M' for i in size_unit] # find all indexes of MB
  data.loc[kb_loc ,'Size'] = [pd.to_numeric(kb[:-1])/1024.0 for kb in data.loc[kb_loc, 'Size']]
  data.loc[mb_loc ,'Size'] = pd.to_numeric([mb[:-1] for mb in data.loc[mb_loc ,'Size']])

  def fotmat_install(d):
    return d.replace(',','').replace('+','')
  data.Installs = pd.to_numeric([fotmat_install(i) for i in data.Installs])
```

Excellent! Now the data is ready and we can start visualizing the Android market!

# Data Visualization


## Size and Ratings
To begin with size and ratings:

<img src="/images/android_analysis/Figure_1.jpg" alt="Figure 1">
<img src="/images/android_analysis/Figure_2.jpg" alt="Figure 2">

The average app file size for android is 22 MB. However, this is due to a small number of apps with large file size. Majority of the app size are clustered in 0 - 10 MB, with 4MB being the most common. That is also the reason why the plot is skewing to the right.

On the other hand, User Rating has a left-skewed distribution. In fact, among all valid ratings, less than 4% rated below or equal 3. Majority of the ratings are between 4.0 and 5.0, meaning people generally give high ratings to apps.

Let's see how different categories have an effect in app size.

<img src="/images/android_analysis/Figure_3.jpg" alt="Figure 3">

There are 33 categories in the dataset. For simplicity, I am only show the top and the bottom 5 categories. As we can see in the boxplot, games generally has larger file size(median at around 40), while libraries and demo has the smallest size (median less than 4).

<img src="/images/android_analysis/Figure_4.jpg" alt="Figure 4">

Rating do not seem to differ much in categories, but we definitely see a difference when we compare the top rating app category (Health & Fitness) and the worst rating category (Dating). The median difference between the two is only 0.4.

## Free vs. Paid

<img src="/images/android_analysis/Figure_5.jpg" alt="Figure 5">

Almost 93% of the apps in the android market are free, and the average rating between the two are similar. People seem to favor free app over paid app.

## Category

<img src="/images/android_analysis/Figure_6.jpg" alt="Figure 6">
<img src="/images/android_analysis/Figure_7.jpg" alt="Figure 7">

It is interesting that 'Family' is only rank 6th in the number of installs while it actually has almost doubled the number of apps of 'Game'. It shows that app developers or companies maybe on the wrong focus since android users seem to think other categories such as 'Communication' and 'Productivity' apps are more important than 'Family' apps.

## Game Genres

Suppose a game design company is developing a game that targets to attract millions players on Android, they are open-minded and are willing to accept any genre as long as the game will be popular.

#### What game genre should they develop?

 Let's get an idea about what other game design companies think of game genres:
<img src="/images/android_analysis/Figure_8.jpg" alt="Figure 8">

Wow, 'Action' game almost hold one third of the gaming apps, with over 32%, while simulation only hold about 1% of the market. If 'Action' games are so popular, then 'Action' Games must be the way to go! But wait, this is only parts of the story. We need to consider what consumers think about the market too. We can accomplish this by understanding the number of installs by different genres.

<img src="/images/android_analysis/Figure_9.jpg" alt="Figure 9">

It turns out that 'Action' games are not the most popular game in Android market, according to the number of installs by game genre, which ranks 2nd, while 'Arcade' holds the top spot. Even though Arcade games are popular among all game genres, I would not develop an Arcade game if I were the game design company. 20% of the market share in games are Arcade. It means that there is an Arcade game for every five games. It is difficult to stand out from the crowd when you have many other competitors in the same genre. 'Casual' games, on the other hand, shows a great potential as it held about 20% of the market shares in installs, while less than 5% of the game are 'Casual' in the market right now. # It means that people likes to play causal games and currently they are not as saturated as Action games and Arcade games.

#### What ratings should I be aiming for if I am developing games?

<img src="/images/android_analysis/Figure_10.jpg" alt="Figure 10">

According to this boxplot, most of the game genre has a median of 4.25. Depending on which genre the company choose, a more concise boxplot is available for each genre. For example for 'Casual' Games, we should be expecting a rating of 4.4 (lower quartile) and aiming for 4.5 (upper quartile) if the company want the game to succeed. 'Casual' Game also has a smaller variance comparing other game genres like 'Racing' and 'Board' games, which benefits us as we understand that players in 'Casual' game generally is pretty satisfied with the game.

## Customer Reviews

Reviews are important for software companies, as they are one of the most straight forward way to listen from the users. In fact, people do not leave comments or reviews for nothing. they write reviews because they have something to praise or complain about the app. That is why reviews are so important for any companies. As a result, Companies should definitely pay attention to those reviews. Lets begin with checking the number of reviews by category.

<img src="/images/android_analysis/Figure_11.jpg" alt="Figure 11">

This graph is not very informative as 'Game' takes over the reviews. It is actually expected given the fact that 'games' has a much larger numbers in installs compared to less popular category like 'Beauty' and 'Events' The story would be clearer if we divide them by the number of installs according to their category. This way we are able to understand how likely a user writes a review in a given category.

<img src="/images/android_analysis/Figure_12.jpg" alt="Figure 12">

This plot shows a surprising result. 'Comics' actually holds the top spot with 6%, meaning people who downloaded 'Comics' tends to write more reviews compared to other app categories. Categories that has the least likelihood to have reviews are 'News & Magazines', 'Productivity', 'Travel & Local', and 'Events'. They all have less than 1% of the users to write reviews. They will probably need to find another way to get feedback regarding the performance of the apps.


# Conclusion

In this post, I demonstrated how much we can analyze using data from Android Market. We explored how categories affect the average size of an app, how a game design company would use Android Market data to analyze current gaming market, how satisfy users feel for different app categories, and what categories should and shouldn't pay extra attention to reviews. There are many more questions can be answered in this dataset. It is up to you to decide how to utilize it and unleash it's potential.

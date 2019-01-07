---
layout: archive
permalink: /test-page/
title: "How do Machine Learning Algorithms work?"
author_profile: true
header:
  image: "/images/ml_chart.jpg"  
mathjax: "true"
---
# Logistic Regression
Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function. ([Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression))

## How does logistic regression work?
Let's consider a scenario in which we are predicting whether a person has diabetes based on their age and weight. We first plug the predictors into a linear function: $$L=b_0+b_1*age+b_2*weight$$. Here we called L as log-odds, which is a linear combination of age and weight, with $$b_0$$ as a constant, and $$b_1$$, $$b_2$$ as coefficients.
We then transform the log-odds to probabilities using the logistic regression formula: $$p=e^L/(1+e^L)$$, where $$p$$ is probability. This formula gives us a S-shaped probability distribution. We then apply log to all data points after plugging into the formula above. Adding them up gives us their likelihood of that particular S-shaped function. We then rotate the log odds line and repeat the above process in order to find the maximum likelihood.

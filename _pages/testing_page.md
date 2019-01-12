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

## Characteristics of Logistic regression

 - Widely used in many classification problems
 - Output can be interpreted as probability
 - Perform well even in a small dataset
 - Cannot solve non-linear problem

## How does logistic regression work?
Let's consider a scenario in which we are predicting whether a person has diabetes based on their age and weight.

We first plug the predictors into a linear function: $$L=b_0+b_1*age+b_2*weight$$. We called L as log-odds, which is a linear combination of age and weight, with $$b_0$$ as a constant, and $$b_1 & b_2$$ as coefficients. We then transform the log-odds to probabilities using the logistic regression formula: $$p=e^L/(1+e^L)$$.

<img src="/images/ml/image2.jpg" alt="lr1">

This formula gives us a S-shaped probability distribution, with the range 0 to 1. Therefore, we can interpret $$p$$ as probability. We then apply log to all data points after plugging into the formula above. Adding them up gives us their likelihood of that particular S-shaped function. The best model can be achieved when we maximize the likelihood. In order to accomplish the maximum likelihood, we keep rotating the log odds line and repeat the calculation above.

Therefore, logistic regression is more like a linear classifier where it calculates a score (probability) to predict whether it belongs to 0 or 1.


# Decision Trees

Decision trees breakdown a dataset into smaller subsets (decision nodes, chance nodes, and end nodes). Decision nodes indicate a decision to be made, and chance nodes are the children of decision nodes, which shows the probability of certain results. End nodes represent the outcome of the model. Decision trees are built using a technique called ID3, which use entropy and information gain to construct a decision tree.

## Definition

Before we explore how decision trees works, we need to understand the following terms:
 - Purity: how well two (or more) classes are separated after the split.
 - Entropy: It indicates how pure one subset is. In other words, it tells you how un-certain you are about a randomly picked item is yes or no. We want to *minimize entropy* as much as possible. We will explore further below in detail.
 - Information gain: It measures how useful a feature is for splitting the label (yes/no). We want to *maximize information gain*, and the feature with the highest information gain will split first.

## How does decision trees work?

There are many ways to measure the purity of the split. One way is utilizing entropy and information gain.
Entropy: $$H(S) = -p_1*log2 p_1 – p_0 log2 p_0$$, where S is the subset of training examples, $$p_1$$ is the % of positive examples in S, $$p_0$$ is the % of negative examples in S.

<img src="/images/ml/image1.jpg" alt="table1">

Consider the table above, if we do NOT ask peers, there will be 4 possible outcomes: 2 Yes (indicating pass) and 2 No (indicating no pass). The entropy, H(S), then would be $$-0.5*log_2 0.5-0.5*log_2 0.5=1$$ bit. On the other hand, if we watch video, we will be have 3 possible outcomes that are all yes. H(S) would be $$1*log_2 1 - 0*log_2 0=0$$ bit.

(get a entropy graph.)
Information gain
Information gain measures how useful a feature is for splitting the label (yes/no). We want to maximize information gain, and the feature with the highest information gain will split first.
is the expected reduction in entropy due to sorting on a particular attribute (say an attribute that has small entropy)

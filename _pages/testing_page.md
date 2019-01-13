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

We first plug the predictors into a linear function: $$L=b_0+b_1 age+b_2 weight$$. We called L as log-odds, which is a linear combination of age and weight, with $$b_0$$ as a constant, and $$b_1, b_2$$ as coefficients. We then transform the log-odds to probabilities using the logistic regression formula: $$p=e^L/(1+e^L)$$.

<img src="/images/ml/image2.jpg" alt="lr1">

This formula gives us a S-shaped probability distribution, with the range 0 to 1. Therefore, we can interpret $$p$$ as probability. We then apply log to all data points after plugging into the formula above. Adding them up gives us their likelihood of that particular S-shaped function. The best model can be achieved when we maximize the likelihood. In order to accomplish the maximum likelihood, we keep rotating the log odds line and repeat the calculation above.

Therefore, logistic regression is more like a linear classifier where it calculates a score (probability) to predict whether it belongs to 0 or 1.


# Decision Trees

Decision trees breakdown a dataset into smaller subsets (decision nodes, chance nodes, and end nodes). Decision nodes indicate a decision to be made, and chance nodes are the children of decision nodes, which shows the probability of certain results. End nodes represent the outcome of the model. Decision trees are built using a technique called ID3, which use entropy and information gain to construct a decision tree.

## Definition

Before we explore how decision trees works, we need to understand the following terms:
 - Purity: how well two (or more) classes are separated after the split.
 - Entropy: It indicates how pure one subset is. In other words, it tells you how un-certain you are about a randomly picked item is yes or no. We want to **minimize entropy** as much as possible. We will explore further below in detail.
 - Information gain: It measures how useful a feature is for splitting the label (yes/no). We want to **maximize information gain**, and the feature with the highest information gain will split first.

## Characteristics of Decision Trees
 - Easy to understand and explain
 - Can handle non-linear features
 - Prone to overfitting
 - cannot train unseen categorical values

## How does decision trees work?

There are many ways to measure the purity of the split. One way is utilizing entropy and information gain.
Entropy: $$H(S) = -p_1 log_2 p_1 – p_0 log_2 p_0$$, where $$S$$ is the subset of training examples, $$p_1$$ is the % of positive examples in $$S$$, $$p_0$$ is the % of negative examples in $$S$$.

<img src="/images/ml/image1.jpg" alt="table1">

Consider the table above, if we do NOT ask peers, there will be 4 possible outcomes: 2 Yes (indicating pass) and 2 No (indicating no pass). The entropy, $$H(S)$$, then would be $$-0.5 log_2 0.5-0.5 log_2 0.5=1$$ bit. On the other hand, if we watch video, we will be have 3 possible outcomes, which are all yes. $$H(S)$$ would be $$1 log_2 1 - 0 log_2 0=0$$ bit.

Information gain

let A = playing video games, and B = not playing video games.
The information gain of playing games is: $$Gain(S, game)= H(S) - ( p(A)H(S_A)+p(B)H(S_B) )$$. Here H(S) = 1 because there are 6 possible outcomes, 3 yes and 3 no. The probability of playing games is 3/6, and the probability of not playing games is 3/6. The entropy of playing games is $$-0.33 log_2 0.33-0.67 log_2 0.67=0.91$$, and entropy for not playing games is $$-0.67 log_2 0.67-0.33 log_2 0.33=0.91$$. Therefore, $$Gain(S, "game")= 1 - ( 0.5*0.91+0.5*0.91 )=0.09$$.

 $$Gain(S, "video")= 1 - ( 1*0+0*0 )=1$$.
 $$Gain(S, "Ask peers")= 1 - ( 2/6*1+4/6*1 )=0$$.

Since we are trying to maximum the information gain, the predictor we split first for predicting whether we are going to pass or not is watching tutorial videos (yes/no).

# Naïve Bayes

Naïve Bayes is a simple yet powerful machine learning algorithm that does a great job in supervised learning. It uses a probability theory, Bayes’ theorem, to make classifications.

## Characteristics of Naïve Bayes
 - generally useful in NLP (text) problems
 - Easy and fast to make predictions
 - performs well in multi-class predictions
 - Assuming each feature is independence of one another
 - Performs well when inputs are categorical variables
 - If a variable in the test set has a new value that is never observed in the training phrase, it will return 0 probability (although can be solved with some techniques)

## How does Naïve Bayes work?

<img src="/images/ml/image3.jpg" alt="table2">

Consider the table above, if we have a new statement "Help my dad is missing", how does Naïve Bayes decide whether it is emergency or non-emergency statement?

**word frequencies**. Based on how many times a word occurs in a sentence, given its label, we can get a sense of how it belongs to one label or not using Bayes Theorem: $$P(A|B)= P(B|A)P(A)/P(B)$$.

Let’s say A is an Emergency statement, and B is the text sentence we are interested in testing. **By assuming each word is independent of one another**, $$P(help my son is missing) = P(help) * P(my) * P(son) * P(is) * P(missing)$$, where P indicates probability.

We now have every info to calculate the probability for this problem; starting with $$P(A)$$, the probability of an emergency statement, appeared three out of five times, therefore $$P(A)=3/5$$.

$$P(help)$$ is the frequency of help out of all word frequency: $$1/(4+5+3+6+6) = 1/24$$, $$P(my) = 2/24$$, $$P(dad) = 2/24$$, $$P(is) = 1/24$$, $$P(missing) = 1/24$$.

P(help|Emergency) means the probability of the frequency of "help" appeared in Emergency label: $$1/13$$, $$P(my|Emergency) = 1/13$$, $$P(dad|Emergency) = 1/13$$, $$P(is|Emergency) = 0/13$$, $$P(missing) = 1/13$$.

Then the equation becomes:

P(Emergency|“help my dad is missing”) = $$P(help|Emergency)P(my|Emergency)P(dad|Emergency)P(is|Emergency)P(missing|Emergency)P(Emergency)/ (P(help)P(my)P(dad)P(is)P(missing)$$

Note that P(is|Emergency) = 0, which will mess up the calculation as everything multiple by 0 is 0. To solve this issue, we need to apply **Laplace smoothing**, which add **all unique words by 1**. That is, to add 19 to the denominator, and add 1 to the numerator. E.g. $$P(help|Emergency) = 2/32$$.

Then the equation becomes:
$$P(Emergency|"help my dad is missing") = (2/32*2/32*2/32*1/32*2/32)(3/5)/ (2/43*3/43*3/43*2/43*3/43) = 0.389$$

$$P(Non-Emergency|"help my dad is missing") = (1/32*3/32*2/32*2/32*1/32)(2/5)/ (2/43*3/43*3/43*2/43*3/43) = 0.173$$

As a result, the probability of being an emergency statement is 0.389, and being an non-emergency statement is 0.173.
In addition, the likelihood of the event, or likelihood of being an emergency statement , is 0.389/(0.389+0.173) = .692. Since 0.692 is higher than 0.5, the model will label this sentence as emergency statement.

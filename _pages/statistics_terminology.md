---
layout: archive
permalink: /statistics_term/
title: "Statistics Terminology"
author_profile: true
header:
  image: "/images/HK2.jpg"  
---

# Hypothesis Testing
Hypothesis testing is a method that compares two or more data sets, or groups. There are two terminologies that one should be familiarized when referring to hypothesis testing: **null hypothesis** and **alternative hypothesis**.
- Null hypothesis represents the value of one variable in group A is the same as the value of the same variable in group B.
- Alternative hypothesis means the value of one variable in group A is not equal to the value of the same variable in group B. Instead of "not equal", it can also be greater than or less than, depending on how to set it up.

## Hypothesis Testing Example: Coffee influences attention?
<img src="/images/coffee.jpg" alt="coffee">

For example, we are interested in knowing whether coffee influences people's focus when completing a task, especially the ones who have started this habit for years. Let's assume we gathered 60 participants that drink coffee daily, 30 of them are not allowed to drink coffee for the day of the experiment (say group A), and the other 30 are given a coffee prior the experiment (say group B). For the experiment, they are given a picture: a typical "where's Wally" picture, and are asked to find where Wally is in the picture. The time is measured in seconds.

In this case, the null hypothesis would be the average time of group A for finding Wally is the same as the average time of group B. In other words, null hypothesis represents coffee does not influence people's focus when completing a task. It is represented as: H<sub>0</sub>: &mu;<sub>A</sub> = &mu;<sub>B</sub>, where &mu is the average time for completing the task.

The alternative hypothesis in this case could be there is a difference between the time of locating wally for group A and group B, usually represented as H<sub>a</sub>: &mu;<sub>A</sub> ≠ &mu;<sub>B</sub>.

We would then run a statistical test and calculate whether there is a strong evidence, or significance, of disproving the null hypothesis. In this case, running two-sample t-test makes more sense because there are only two groups (group A and group B) and interested in whether two groups truly have difference in completion time.

# Linear Regression
Linear regression attempts to model the relationship between predictors variables to a response variable. For example, there can be a linear relationship between income and expense. The wealthier you are, the more money you are going to spend.

Work in progress.

# p-value
P-value is a metric we used to determine whether we can reject the **null hypothesis** in a statistical test. It is the probability of the null hypothesis being true. In other words, the smaller the p value, the stronger evidence we have to reject the null hypothesis.

# Confidence Interval
Work in progress.

---
layout: archive
permalink: /test_page/
title: "How do Machine Learning Algorithms work?"
author_profile: true
header:
  image: "/images/ml_chart.jpg"  
---
# Logistic Regression
Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function. ([Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression))

Let's consider a scenario in which we are predicting whether a person has diabetes based on their age and weight. We first plug the predictors into a linear function. L = B_0 + B_1*age + B2*weight. Here we called L as log-odds, which is a linear combindation of age and weight, with a constant B_0, and B_1 and B_2 are coefficients.
We then transform the L to probabilities using the following formula: p= e^L / (1+e^L), where p is probability. This formula gives us a S-shaped probability distribution. We then apply log to all data points after plugging into the S-shaped distribution. The sum of them are their likelihood of that particular S-shaped function. We then rotate the log odds line and transform it to probabilities and find the maxmium likelihood.

<img src="/images/voice_recognition/Figure_3.jpg" alt="Figure 3">


$$z=x+y$$
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
  $$L=alpha_0+beta_0$$
</script>

tt

---
title: summary of Improving-Deep-Neural-Networks
date: 2018-03-02
copyright: true
categories: English
tags: [Improving Deep Neural Networks, deep learning]
mathjax: true
mathjax2: true
toc: true
top: true
---

## Note
This is my personal summary after studying the course, [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/), which belongs to Deep Learning Specialization. and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

## My personal notes
${1_{st}}$ week: [practical-aspects-of-deep-learning](/2018/03/01/01_practical-aspects-of-deep-learning/)
-  [01_setting-up-your-machine-learning-application](/2018/03/01/01_practical-aspects-of-deep-learning/#01_setting-up-your-machine-learning-application)
  	- [01_train-dev-test-sets](/2018/03/01/01_practical-aspects-of-deep-learning/#01_train-dev-test-sets)
  	- [02_bias-variance](/2018/03/01/01_practical-aspects-of-deep-learning/#02_bias-variance)
  	- [03_basic-recipe-for-machine-learning](/2018/03/01/01_practical-aspects-of-deep-learning/#03_basic-recipe-for-machine-learning)
- [02_regularizing-your-neural-network](/2018/03/01/01_practical-aspects-of-deep-learning/#02_regularizing-your-neural-network)
	- [01_regularization](/2018/03/01/01_practical-aspects-of-deep-learning/#01_regularization)
	- [02_why-regularization-reduces-overfitting](/2018/03/01/01_practical-aspects-of-deep-learning/#02_why-regularization-reduces-overfitting)
	- [03_dropout-regularization](/2018/03/01/01_practical-aspects-of-deep-learning/#03_dropout-regularization)
	- [04_understanding-dropout](/2018/03/01/01_practical-aspects-of-deep-learning/#04_understanding-dropout)
	- [05_other-regularization-methods](/2018/03/01/01_practical-aspects-of-deep-learning/#05_other-regularization-methods)
- [03_setting-up-your-optimization-problem](/2018/03/01/01_practical-aspects-of-deep-learning/#03_setting-up-your-optimization-problem)
	- [01_normalizing-inputs](/2018/03/01/01_practical-aspects-of-deep-learning/#01_normalizing-inputs)
	- [02_vanishing-exploding-gradients](/2018/03/01/01_practical-aspects-of-deep-learning/#02_vanishing-exploding-gradients)
	- [03_weight-initialization-for-deep-networks](/2018/03/01/01_practical-aspects-of-deep-learning/#03_weight-initialization-for-deep-networks)
	- [04_numerical-approximation-of-gradients](/2018/03/01/01_practical-aspects-of-deep-learning/#04_numerical-approximation-of-gradients)
	- [05_gradient-checking](/2018/03/01/01_practical-aspects-of-deep-learning/#05_gradient-checking)
	- [06_gradient-checking-implementation-notes](/2018/03/01/01_practical-aspects-of-deep-learning/#06_gradient-checking-implementation-notes)

$2_{nd}$ week: [optimization-algorithms](/2018/03/02/02_optimization-algorithms/)
- [01_mini-batch-gradient-descent](/2018/03/02/02_optimization-algorithms/#01_mini-batch-gradient-descent)
- [02_understanding-mini-batch-gradient-descent](/2018/03/02/02_optimization-algorithms/#02_understanding-mini-batch-gradient-descent)
- [03_exponentially-weighted-averages](/2018/03/02/02_optimization-algorithms/#03_exponentially-weighted-averages)
- [04_understanding-exponentially-weighted-averages](/2018/03/02/02_optimization-algorithms/#04_understanding-exponentially-weighted-averages)
- [05_bias-correction-in-exponentially-weighted-averages](/2018/03/02/02_optimization-algorithms/#05_bias-correction-in-exponentially-weighted-averages)
- [06_gradient-descent-with-momentum](/2018/03/02/02_optimization-algorithms/#06_gradient-descent-with-momentum)
- [07_rmsprop](/2018/03/02/02_optimization-algorithms/#07_rmsprop)
- [08_adam-optimization-algorithm](/2018/03/02/02_optimization-algorithms/#08_adam-optimization-algorithm)
- [09_learning-rate-decay](/2018/03/02/02_optimization-algorithms/#09_learning-rate-decay)
- [10_the-problem-of-local-optima](/2018/03/02/02_optimization-algorithms/#10_the-problem-of-local-optima)

$3_{rd}$ week: [hyperparameter-tuning-batch-normalization-and-programming-frameworks](/2018/03/02/03_hyperparameter-tuning-batch-normalization-and-programming-frameworks/)
- [01_hyperparameter-tuning](/2018/03/02/02_optimization-algorithms/#01_hyperparameter-tuning)
  -   [01_tuning-process](/2018/03/02/02_optimization-algorithms/#01_tuning-process)
  -   [02_using-an-appropriate-scale-to-pick-hyperparameters](/2018/03/02/02_optimization-algorithms/#02_using-an-appropriate-scale-to-pick-hyperparameters)
  -   [03_hyperparameters-tuning-in-practice-pandas-vs-caviar](/2018/03/02/02_optimization-algorithms/#03_hyperparameters-tuning-in-practice-pandas-vs-caviar)
- [02_batch-normalization](/2018/03/02/02_optimization-algorithms/#02_batch-normalization)
  -   [01_normalizing-activations-in-a-network](/2018/03/02/02_optimization-algorithms/#01_normalizing-activations-in-a-network)
  -   [02_fitting-batch-norm-into-a-neural-network](/2018/03/02/02_optimization-algorithms/#02_fitting-batch-norm-into-a-neural-network)
  -   [03_why-does-batch-norm-work](/2018/03/02/02_optimization-algorithms/#03_why-does-batch-norm-work)
  -   [04_batch-norm-at-test-time](/2018/03/02/02_optimization-algorithms/#04_batch-norm-at-test-time)
- [03_multi-class-classification](/2018/03/02/02_optimization-algorithms/#03_multi-class-classification)
  - [01_softmax-regression](/2018/03/02/02_optimization-algorithms/#01_softmax-regression)
  - [02_training-a-softmax-classifier](/2018/03/02/02_optimization-algorithms/#02_training-a-softmax-classifier)
- [04_introduction-to-programming-frameworks](/2018/03/02/02_optimization-algorithms/#04_introduction-to-programming-frameworks)
  - [tensorflow](/2018/03/02/02_optimization-algorithms/#02_tensorflow)

## My personal programming assignments
week1: [practical-aspects-of-deep-learning](/2018/03/01/practical-aspects-of-deep-learning/)
week2: [optimization-algorithms](/2018/03/02/OptimizationMethods/)
week3: [hyperparameter-tuning-batch-normalization-and-programming-frameworks](/Tensorflow%20Tutorial/)

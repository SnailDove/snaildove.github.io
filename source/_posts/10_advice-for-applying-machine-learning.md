---
title: 10_advice-for-applying-machine-learning note10
date: 2018-01-10
copyright: true
categories: English
tags: [Machine Learning]
mathjax: true
mathjax2: true
---

By now you have seen a lot of different learning algorithms. And if you've been following along these videos you should consider yourself an expert on many state-of-the-art machine learning techniques. **But even among people that know a certain learning algorithm. There's often a huge difference between someone that really knows how to powerfully and effectively apply that algorithm, versus someone that's less familiar with some of the material that I'm about to teach and who doesn't really understand how to apply these algorithms and can end up wasting a lot of their time trying things out that don't really make sense.**

What I would like to do is make sure that if you are developing machine learning systems, that you know how to choose one of the most promising avenues to spend your time pursuing. And on this and the next few videos I'm going to give a number of practical suggestions, advice, guidelines on how to do that. 

And concretely what we'd focus on is the problem of, suppose you are developing a machine learning system or trying to improve the performance of a machine learning system, how do you go about deciding what are the proxy avenues to try next? 

To explain this, let's continue using our example of learning to predict housing prices. And let's say you've implement and regularize linear regression. Thus minimizing that cost function j.  Now suppose that after you take your learn parameters, if you test your hypothesis on the new set of houses, suppose you find that this is making huge errors in this prediction of the housing prices. The question is what should you then try mixing in order to improve the learning algorithm? 

There are many things that one can think of that could improve the performance of the learning algorithm. 

One thing they could try, is to get more training examples. And concretely, you can imagine, maybe, you know, setting up phone surveys, going door to door, to try to get more data on how much different houses sell for. And the sad thing is I've seen a lot of people spend a lot of time collecting more training examples, thinking oh, if we have twice as much or ten times as much training data, that is certainly going to help, right? **But sometimes getting more training data doesn't actually help and in the next few videos we will see why, and we will see how you can avoid spending a lot of time collecting more training data in settings where it is just not going to help.** 

Other things you might try are to well maybe try a smaller set of features. So if you have some set of features such as x1, x2, x3 and so on, maybe a large number of features. Maybe you want to spend time carefully selecting some small subset of them to prevent overfitting. Or maybe you need to get additional features. Maybe the current set of features aren't informative enough and you want to collect more data in the sense of getting more features. 

And once again this is the sort of project that can scale up the huge projects can you imagine getting phone surveys to find out more houses, or extra land surveys to find out more about the pieces of land and so on, so a huge project. And once again it would be nice to know in advance if this is going to help before we spend a lot of time doing something like this. We can also try adding polynomial features things like x2 square x2 square and product features x1, x2. We can still spend quite a lot of time thinking about that and we can also try other things like decreasing lambda, the regularization parameter or increasing lambda. Given a menu of options like these, some of which can easily scale up to six month or longer projects. 

**Unfortunately, the most common method that people use to pick one of these is to go by** ***gut feeling***. In which what many people will do is sort of randomly pick one of these options and maybe say, "Oh, lets go and get more training data." And easily spend six months collecting more training data or maybe someone else would rather be saying, "Well, let's go collect a lot more features on these houses in our data set." And I have a lot of times, sadly seen people spend, you know, literally 6 months doing one of these avenues that they have sort of at random only to discover six months later that that really wasn't a promising avenue to pursue. 

**Fortunately, there is a pretty simple technique that can let you very quickly rule out half of the things on this list as being potentially promising things to pursue.** And there is a very simple technique, that if you run, can easily rule out many of these options, and potentially save you a lot of time pursuing something that's just is not going to work. 

In the next two videos after this, I'm going to first talk about **how to evaluate learning algorithms**. And in the next few videos after that, I'm going to talk about these techniques, which are called **the machine learning diagnostics**. And what a diagnostic is, is a test you can run, to get insight into what is or isn't working with an algorithm, and which will often give you insight as to what are promising things to try to improve a learning algorithm's performance. We'll talk about specific diagnostics later in this video sequence. But I should mention in advance that diagnostics can take time to implement and can sometimes, you know, take quite a lot of time to implement and understand but doing so can be a very good use of your time when you are developing learning algorithms because they can often save you from spending many months pursuing an avenue that you could have found out much earlier just was not going to be fruitful. So in the next few videos, I'm going to first talk about how evaluate your learning algorithms and after that I'm going to talk about some of these diagnostics which will hopefully let you much more effectively select more of the useful things to try mixing if your goal to improve the machine learning system.

# 01_evaluating-a-learning-algorithm

## Evaluating a Hypothesis

Once we have done some trouble shooting for errors in our predictions by:

- Getting more training examples
- Trying smaller sets of features
- Trying additional features
- Trying polynomial features
- Increasing or decreasing λ

  We can move on to evaluate our new hypothesis.

A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a **training set** and a **test set** . Typically, the training set consists of 70 % of your data and the test set is the remaining 30 %.


The new procedure using these two sets is then:

1. Learn $\Theta$ and minimize $J_{train}(\Theta)$  using the training set
2. Compute the test set error $J_{test}(\Theta)$

The test set error For linear regression: $J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2$


For classification ~ Misclassification error (aka 0/1 misclassification error):  

 $$err(h_\Theta(x),y) = \begin{cases} 1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline 0 & \mbox otherwise \end{cases}$$


This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

$$\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})$$

This gives us the proportion of the test data that was misclassified.

## Model Selection and Train/Validation/Test Sets

Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could ***overfit*** and as a result your predictions on the test set would be poor. ***The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.***

**Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.** 

One way to break down our dataset into the three sets is:  

- ​    Training set: 60%   
- ​    Cross validation set: 20%   
- ​    Test set: 20%   

We can now calculate three separate error values for the three different sets using the following method: 

1. ​    Optimize the parameters in $Θ $ using the training set for each polynomial degree.   
2. ​    Find the polynomial degree d with the least error using the cross validation set.   
3. ​    Estimate the generalization error using the test set with $J_{test}(Θ^{(d)})$, (d = theta from polynomial with lower error);   

**This way, the degree of the polynomial d has not been trained using the test set.**

*Training error:*
$$J_{train}\left(\theta\right) = \frac{1}{2m}\sum_\limits{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^2$$
*Cross Validation error:*
$$J_{cv}\left(\theta\right) = \frac{1}{2m_{cv}}\sum_\limits{i=1}^{m}\left(h_{\theta}\left(x^{(i)}_{cv}\right)-y^{(i)}_{cv}\right)^2$$
*Test error:*
$$J_{cv}\left(\theta\right) = \frac{1}{2m_{cv}}\sum_\limits{i=1}^{m}\left(h_{\theta}\left(x^{(i)}_{cv}\right)-y^{(i)}_{cv}\right)^2$$

# 02_bias-vs-variance

## Diagnosing Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

- We need to distinguish whether **bias** or **variance** is the problem contributing to bad predictions.   
- High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.   

![img](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/1.png)

The training error will tend to **decrease** as we increase the degree d of the polynomial. 

At the same time, the cross validation error will tend to **decrease** as we increase d up to a point, and then it will **increase** as d is increased, forming a convex curve. 

![high_bias_and_high_variance](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/2.png)



**High bias (underfitting)**: both $J_{train}(Θ)$ and $J_{CV}(Θ)$ will be high. Also, $J_{CV}(Θ)≈J_{train}(Θ)$. 

**High variance (overfitting)**: $J_{train}(Θ)$ will be low and $J_{CV}(Θ)$ will be much greater than $J_{train}(Θ)$. 

The is summarized in the figure below: 

![img](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/3.png)

## Regularization and Bias/Variance

**Note:**    [The regularization term below and through out the video should be $\frac \lambda {2m} \sum _{j=1}^n \theta_j ^2$ and     **NOT**    $\frac \lambda {2m} \sum _{j=1}^m \theta_j ^2$]  

![3_kinds_of_Linear_regression_with_regularization](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/4.png)

![img](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/5.png)

In the figure above, we see that as $\lambda$ increases, our fit becomes more rigid. On the other hand, as $\lambda$ approaches 0, we tend to over overfit the data. So how do we choose our parameter $\lambda$ to get it 'just right' ? In order to choose the model and the regularization term $λ$ , we need to: 

![img](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/6.png)

1.  Create a list of lambdas (i.e.$ λ∈\{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24\}$);   
2.  Create a set of models with different degrees or any other variants.   
3.  ***Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$ .***
4.  Compute the cross validation error using the learned $Θ$ (computed with $λ$) on the $J_{CV}(\Theta)$ **without** regularization or $λ = 0$.   
5.  Select the best combo that produces the lowest error on the cross validation set.   
6.  Using the best combo $Θ$ and λ, apply it on $J_{test}(\Theta)$ to see if it has a good generalization of the problem.

## Learning Curves


Training an algorithm on a very few number of data points (such as 1, 2 or 3) will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. Hence:

As the training set gets larger, the error for a quadratic function increases.

The error value will plateau out after a certain m, or training set size.

**Experiencing high bias:**

**Low training set size**  : causes $J_{train}(\Theta)$ to be low and $J_{CV}(\Theta)$ to be high. 

**Large training set size** : causes both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ to be high with $J_{train}(\Theta)$≈$J_{CV}(\Theta)$.

If a learning algorithm is suffering from **high bias**, getting more training data will not **(by itself)** help much.

![high_bias_on_training_size](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/7.png)

**Experiencing high variance:**


**Low training set size** : $J_{train}(\Theta)$ will be low and $J_{CV}(\Theta)$ will be high.

**Large training set size**: $J_{train}(\Theta)$ increases with training set size and $J_{CV} (\Theta$) continues to decrease without leveling off. Also, $J_{train}(\Theta)$ < $J_{CV}(\Theta)$ but the difference between them remains significant.

If a learning algorithm is suffering from **high variance**, getting more training data is likely to help.

![high_variance_on_training_size](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/8.png)

## Deciding What to Do Next Revisited

 Our decision process can be broken down as follows: 

- ​         **Getting more training examples:** Fixes high variance   


- ​         **Trying smaller sets of features:** Fixes high variance   


- ​         **Adding features:** Fixes high bias   


- ​         **Adding polynomial features:** Fixes high bias   


- ​         **Decreasing λ:**  Fixes high bias   


- ​         **Increasing λ:**  Fixes high variance.   

##  Diagnosing Neural Networks

- A neural network with fewer parameters is **prone to underfitting**. It is also **computationally cheaper**. 
- A large neural network with more parameters is **prone to overfitting**. It is also **computationally expensive**. In this case you can use regularization (increase λ) to address the overfitting.   

![Neural_networks_and_overfitting](http://pwmpcnhis.bkt.clouddn.com/gitpage/ml-andrew-ng/10/9.png)

Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best. 

**Model Complexity Effects:**    

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.   
- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.   
- In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

## Model Selection
Choosing M the order of polynomials.
How can we tell which parameters Θ to leave in the model (known as "model selection")? 
There are several ways to solve this problem: 

* Get more data (very difficult). 
* Choose the model which best fits the data without overfitting (very difficult). 
* Reduce the opportunity for overfitting through regularization .

**Bias: approximation error (Difference between expected value and optimal value)**

* High Bias = UnderFitting (BU) 
* $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ both will be high and $J_{train}(\Theta)$ ≈ $J_{CV}(\Theta)$ 

**Variance: estimation error due to finite data**

* High Variance = OverFitting (VO) 
* $J_{train}(\Theta)$ is low and $J_{CV}(\Theta)$ ≫$J_{train}(\Theta)$ 

**Intuition for the bias-variance trade-off:**

* Complex model => sensitive to data => much affected by changes in X => high variance, low bias. 
* Simple model => more rigid => does not change as much with changes in X => low variance, high bias. 

One of the most important goals in learning: finding a model that is just right in the bias-variance trade-off. 

**Regularization Effects:** 

* Small values of λ allow model to become finely tuned to noise leading to large variance => overfitting. 
* Large values of λ pull weight parameters to zero leading to large bias => underfitting. 

**Model Complexity Effects:**

* Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently. 
* Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance. 
* In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well. 

**A typical rule of thumb when running diagnostics is:**

* More training examples fixes high variance but not high bias. 
* Fewer features fixes high variance but not high bias. 
* Additional features fixes high bias but not high variance. 
* The addition of polynomial and interaction features fixes high bias but not high variance. 
* When using gradient descent, decreasing lambda can fix high bias and increasing lambda can fix high variance (lambda is the regularization parameter). 
* When using neural networks, small neural networks are more prone to under-fitting and big neural networks are prone to over-fitting. Cross-validation of network size is a way to choose alternatives. 

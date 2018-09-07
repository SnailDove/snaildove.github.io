---
title: 07_regularization note7
date: 2018-01-07
copyright: true
categories: english
tags: [Machine Learning by Andrew NG]
mathjax: true
mathjax2: true
---

## Note
This personal note is written after studying the opening course on [the coursera website](https://www.coursera.org), [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

## The Problem of Overfitting

Consider the problem of predicting $y$ from $x ∈ R$. The leftmost figure below shows the result of fitting a $y = θ_0+θ_1x$ to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/1.png)

Instead, if we had added an extra feature $x_2$ , and fit $y=θ_0+θ_1x+θ_2x^2$ , then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a $5^{th}$ order polynomial $y=\sum_{j=0}^{5}\theta_jx^j$ . We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas(x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of  **underfitting** —in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting **. 

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/2.png)

1. **Underfitting, or high bias**, *is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features.*
2. At the other extreme, **overfitting, or high variance**, *is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.* 

**This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:** 

1) Reduce the number of features: 

- ​    Manually select which features to keep.   

- ​    Use a model selection algorithm (studied later in the course, such as PCA).   

  2) Regularization 

- ​    Keep all the features, but reduce the magnitude of parameters $θ_j$.   

- ​    Regularization works well when we have a lot of slightly useful features.

## Cost Function

**Note:**    [5:18 - There is a typo. It should be $\sum_{j=1}^{n} \theta _j ^2$ instead of $\sum_{i=1}^{n} \theta _j ^2$]

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost. 

Say we wanted to make the following function more quadratic:
$$
\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4
$$
We'll want to eliminate the influence of $θ_3x_3$ and $θ_4x_4$ . Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our

**cost function** :
$$
min_\theta\ \dfrac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2
$$
We've added two extra terms at the end to inflate the cost of $θ_3$ and $θ_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $θ_3$ and $θ_4$ to near zero. This will in turn greatly reduce the values of $θ_3x_3$ and $θ_4x_4$ in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms $θ_3x_3$ and $θ_4x_4$.

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/3.png)

We could also regularize all of our theta parameters in a single summation as:
$$
min_\theta\ \dfrac{1}{2m}\  \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2
$$
The λ, or lambda, is the  **regularization parameter** . It determines how much the costs of our theta parameters are inflated. 

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if λ=0 or is too small ?

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/4.png)

## Regularized Linear Regression

**Note:**    [8:43 - It is said that X is non-invertible if m ≤ n. The correct statement should be that X is non-invertible if m < n, and may be non-invertible if m = n. 

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.
$$
J(\theta)=\frac{1}{2m} \left[  \sum\limits_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)^2 + \lambda\sum_{j=1}^{n}\theta_j^2 \right]
$$

### Gradient Descent

We will modify our gradient descent function to separate out θ0θ0 from the rest of the parameters because we do not want to penalize
$$
{% raw %}\begin{align*} & \text{Repeat}\ \lbrace \\ & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\ & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\\ & \rbrace \end{align*}{% endraw %}
$$
The term $\frac{λ}{m} θ_j$ performs our regularization. With some manipulation our update rule can also be represented as:
$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$
The first term in the above equation, $1−α\frac{λ}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $θ_j$ by some amount on every update. Notice that the second term is now exactly the same as it was before.

### Normal Equation

Now let's approach regularization using the alternate method of the non-iterative normal equation. 

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:
$$
{% raw %}\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \\ & \text{where}\ \ L = \begin{bmatrix} 0 & & & & \\ & 1 & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & 1 \\\end{bmatrix}\end{align*}{% endraw %}
$$
L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number λ. 

Recall that **if m < n, then $X^TX$ is non-invertible. However, when we add the term λ⋅L, then $X^TX + λ⋅L$ becomes invertible.(Note: which is proved)**

## Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/5.png)

### Cost Function 

Recall that our cost function for logistic regression was:
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \large]
$$
We can regularize this equation by adding a term to the end:
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$
The second sum, $∑_n^{j=1}θ^2_j$ **means to explicitly exclude** the bias term, $θ_0$. I.e. the θ vector is indexed from 0 to n (holding n+1 values, $θ_0$ through $θ_n$), and this sum explicitly skips $θ_0$, by running from 1 to n, skipping 0. Thus, when computing the equation, we should continuously update the two following equations:

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/6.png)

```python
import numpy as np;

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```

### Advanced optimization

Let's talk about how to get regularized linear regression to work using the more advanced optimization methods. And just to remind you for those methods what we needed to do was to define the function that's called the cost function, that takes us input the parameter vector theta and once again in the equations we've been writing here we used 0 index vectors. So we had theta 0 up to theta N. But because Octave indexes the vectors starting from 1. Theta 0 is written in Octave as theta 1. Theta 1 is written in Octave as theta 2, and so on down to theta N plus 1. And what we needed to do was provide a function. Let's provide a function called cost function that we would then pass in to what we have, what we saw earlier. We will use the `fminunc` and then you know at cost function, and so on, right. But the `fminunc` was function minimization unconstrained in Octave and this will work with `fminunc` was what will take the cost function and minimize it for us. So the two main things that the cost function needed to return were first J-val. And for that, we need to write code to compute the `costfunction` J of theta. 

![](http://p8o3egtyk.bkt.clouddn.com/gitpage/ml-andrew-ng/07/7.png)

Now, when we're using regularized logistic regression, of course the `costfunction` J of theta changes and, in particular, now a cost function needs to include this additional regularization term at the end as well. So, when you compute j of theta be sure to include that term at the end. 

And then, the other thing that this cost function thing needs to derive with a gradient. So gradient one needs to be set to the partial derivative of J of theta with respect to theta zero, gradient two needs to be set to that, and so on. Once again, the index is off by one. Right, because of the indexing from one Octave users. And looking at these terms. This term over here. We actually worked this out on a previous slide is actually equal to this. It doesn't change. Because the derivative for theta zero doesn't change. Compared to the version without regularization. And the other terms do change. And in particular the derivative respect to theta one. We worked this out on the previous slide as well. Is equal to, you know, the original term and then minus londer M times theta 1. Just so we make sure we pass this correctly. And we can add parentheses here. Right, so the summation doesn't extend. And similarly, you know, this other term here looks like this, with this additional term that we had on the previous slide, that corresponds to the gradient from their regularization objective. So if you implement this cost function and pass this into `fminunc` or to one of those advanced optimization techniques, that will minimize the new regularized cost function J of theta. And the parameters you get out will be the ones that correspond to logistic regression with regularization. 

So, now you know how to implement regularized logistic regression. When I walk around Silicon Valley, I live here in Silicon Valley, there are a lot of engineers that are frankly, making a ton of money for their companies using machine learning algorithms. And I know we've only been, you know, studying this stuff for a little while. But if you understand linear regression, the advanced optimization algorithms and regularization, by now, frankly, you probably know quite a lot more machine learning than many, certainly now, but you probably know quite a lot more machine learning right now than frankly, many of the Silicon Valley engineers out there having very successful careers. You know, making tons of money for the companies. Or building products using machine learning algorithms. So, congratulations. You've actually come a long ways. And you can actually, you actually know enough to apply this stuff and get to work for many problems. So congratulations for that. But of course, there's still a lot more that we want to teach you, and in the next set of videos after this, we'll start to talk about a very powerful cause of non-linear classifier. So whereas linear regression, logistic regression, you know, you can form polynomial terms, but it turns out that there are much more powerful nonlinear quantifiers that can then sort of polynomial regression. And in the next set of videos after this one, I'll start telling you about them. So that you have even more powerful learning algorithms than you have now to apply to different problems.



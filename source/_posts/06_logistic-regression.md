---
title: 06_logistic-regression note6
date: 2018-01-06
copyright: true
categories: english
tags: [Machine Learning]
mathjax: true
mathjax2: true
---

## Note
This personal note is written after studying the opening course on [the coursera website](https://www.coursera.org), [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

## Classification

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/1.png)
To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function. 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/2.png)

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification problem** in which $y$ can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and $y$ may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, $y∈\{0,1\}$ . 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+” . Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

## Hypothesis Representation

We could approach the classification problem ignoring the fact that $y$ is discrete-valued, and use our old linear regression algorithm to try to predict $y$ given $x$. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for $h_{θ}(x)$ to take values larger than 1 or smaller than $0$ when we know that $y ∈ \{0, 1\}$. To fix this, let’s change the form for our hypotheses $h_{θ}(x)$ to satisfy $0≤h_{θ}(x)≤1$. This is accomplished by plugging $θ^Tx$ into the **Logistic Function** . 

Our new form uses the **"Sigmoid Function"** , also called the **"Logistic Function"** :
$$
{% raw %}\begin{align*}& h_\theta (x) = g ( \theta^T x ) \\ \\& z = \theta^T x \\& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}{% endraw %}
$$
Using python to implement it :

```python
import numpy as np;
def sigmoid(z):
   return 1 / (1 + np.exp(-z))
```

The following image shows us what the sigmoid function looks like:

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/3.png)

**The function $g(z)$ , shown here, maps any real number to the $(0, 1)$ interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification**. 

**$h_{θ}(x)$ will give us the probability that our output is 1**. For example, $h_{θ}(x)=0.7$ gives us a probability of $70\%$ that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is $70\%$, then the probability that it is 0 is $30\%$).
$$
{% raw %}\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \\ & P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}{% endraw %}
$$

## Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
$$
{% raw %}\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \\ & h_\theta(x) < 0.5 \rightarrow y = 0 \\ \end{align*}{% endraw %}
$$
The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:
$$
{% raw %}\begin{align*}& g(z) \geq 0.5 \\ & when \; z \geq 0\end{align*}{% endraw %}
$$
From these statements we can now say:
$$
{% raw %}\begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \\ & \theta^T x < 0 \Rightarrow y = 0 \\ \end{align*}{% endraw %}
$$
The **decision boundary**  is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function. 

**Example** :
$$
{% raw %}\begin{align*}& \theta = \begin{bmatrix}5 \\ -1 \\ 0\end{bmatrix} \\ & y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0 \\ & 5 - x_1 \geq 0 \\ & - x_1 \geq -5 \\& x_1 \leq 5 \\ \end{align*}{% endraw %}
$$
In this case, our decision boundary is a straight vertical line placed on the graph where $x_1=5$ , and everything to the left of that denotes $y = 1$ , while everything to the right denotes $y = 0$ . 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/4.png)

Again, the input to the sigmoid function g(z) (e.g. $θ^TX$ ) doesn't need to be linear, and could be a function that describes a circle (e.g.  $z=θ_0+θ_1x^2_1+θ_2x^2_2$ ) or any shape to fit our data.

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/5.png)

## Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function. 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/6.png)

Instead, our cost function for logistic regression looks like:
$$
{% raw %}\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}{% endraw %}
$$
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/7.png)
$$
{% raw %}\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \\ & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \\ & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \\ \end{align*}{% endraw %}
$$

1. If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity. 
2. If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity. 

Note that writing the cost function in this way guarantees that $J(θ)$ is convex for logistic regression.

## Simplified Cost Function and Gradient Descent

**Note:**    [6:53 - the gradient descent equation should have a 1/m factor] 

 We can compress our cost function's two conditional cases into one case:
$$
\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))
$$
Notice that when y is equal to 1, then the second term $(1−y)log(1−h_θ(x))$ will be zero and will not affect the result. If y is equal to 0, then the first term $−ylog(h_θ(x))$ will be zero and will not affect the result. 

We can fully write out our entire cost function as follows:

$$
J(\theta) =  \frac{1}{m} \displaystyle \sum_{i=1}^m [-y^{(i)}\log (h_\theta (x^{(i)})) - (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

### A vectorized implementation

$$
{% raw %}\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}{% endraw %}
$$

Using python to implement it :

```python
import numpy as np
def cost(theta, X, y):
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```

### Gradient Descent

Remember that the general form of gradient descent is:
$$
{% raw %}\begin{align*}& Repeat \; \lbrace \\ & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \\ & \rbrace\end{align*}{% endraw %}
$$
We can work out the derivative part using calculus to get:
$$
{% raw %}\begin{align*} & Repeat \; \lbrace \\ & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \\ & \rbrace \end{align*}{% endraw %}
$$
The detail mathematical process : 
$$
{% raw %}\begin{align*} 
J(\theta) &= \frac{1}{m} \displaystyle \sum_{i=1}^m [-y^{(i)}\log (h_\theta (x^{(i)})) - (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))] \\
&= -\frac{1}{m} \displaystyle \sum_{i=1}^m [{{y}^{(i)}}\log ( {h_\theta} {{x}^{(i)}} ) )+ ( 1-{{y}^{(i)}}  )\log  ( 1-{h_\theta} ( {{x}^{(i)}}  )  )] \\
& = -\frac{1}{m} \displaystyle \sum_{i=1}^m [{{y}^{(i)}}\log ( \frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} )+( 1-{{y}^{(i)}})\log ( 1-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} )] \\
& = -\frac{1}{m} \displaystyle \sum_{i=1}^m [ -{{y}^{(i)}}\log  ( 1+{{e}^{-{\theta^T}{{x}^{(i)}}}}  )- ( 1-{{y}^{(i)}}  )\log  ( 1+{{e}^{{\theta^T}{{x}^{(i)}}}}  )] \\
\end{align*}{% endraw %}
$$

$$
{% raw %}\begin{align*}
 \frac{\partial }{\partial {\theta_{j}}}J\left( \theta \right) &= -\frac{1}{m}\frac{\partial }{\partial {\theta_{j}}}[\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^{T}}{{x}^{(i)}}}} \right)]}] \\
& =\frac{\partial }{\partial {\theta_{j}}}[-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^{T}}{{x}^{(i)}}}} \right)]}] \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\frac{-x_{j}^{(i)}{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}{1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}] \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{{y}^{(i)}}\frac{x_j^{(i)}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}] \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}x_j^{(i)}-x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}+{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}} \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}\left( 1\text{+}{{e}^{{\theta^T}{{x}^{(i)}}}} \right)-{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}x_j^{(i)}} \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}} \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}} \\
& =-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}-{h_\theta}\left( {{x}^{(i)}} \right)]x_j^{(i)}} \\
& =\frac{1}{m}\sum\limits_{i=1}^{m}{[{h_\theta}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_j^{(i)}} 
\end{align*}{% endraw %}
$$

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta. 

### A vectorized implementation

$$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$$

### Note

**The idea of feature scaling also applies to gradient descent for logistic regression. And yet we have features that are on different scale, then applying feature scaling can also make grading descent run faster for logistic regression.**

## Advanced Optimization

**Note:**    [7:35 - '100' should be 100 instead. The value provided should be an integer and not a character string.] 

**"Conjugate gradient", "BFGS", and "L-BFGS"** *are more sophisticated, faster ways to optimize $θ$ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.*

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/8.png)

These algorithms actually do more sophisticated things than just pick a good learning rate, and so they often end up converging much faster than gradient descent. These algorithms actually do more sophisticated things than just pick a good learning rate, and so they often end up converging much faster than gradient descent, but detailed discussion of exactly what they do is beyond the scope of this course. **In fact, I actually used to have used these algorithms for a long time, like maybe over a decade, quite frequently, and it was only, you know, a few years ago that I actually figured out for myself the details of what conjugate gradient, BFGS and O-BFGS do. So it is actually entirely possible to use these algorithms successfully and apply to lots of different learning problems without actually understanding the inter-loop of what these algorithms do.** If these algorithms have a disadvantage, I'd say that the main disadvantage is that they're quite a lot more complex than gradient descent. And in particular, you probably should not implement these algorithms - conjugate gradient, L-BGFS, BFGS - yourself unless you're an expert in numerical computing. Instead, just as I wouldn't recommend that you write your own code to compute square roots of numbers or to compute inverses of matrices, for these algorithms also what I would recommend you do is just use a software library. So, you know, to take a square root what all of us do is use some function that someone else has written to compute the square roots of our numbers. And fortunately, Octave and the closely related language MATLAB - we'll be using that - Octave has a very good. Has a pretty reasonable library implementing some of these advanced optimization algorithms. And so if you just use the built-in library, you know, you get pretty good results. I should say that there is a difference between good and bad implementations of these algorithms. And so, if you're using a different language for your machine learning application, if you're using C, C++, Java, and so on, you might want to try out a couple of different libraries to make sure that you find a good library for implementing these algorithms. Because there is a difference in performance between a good implementation of, you know, contour gradient or LPFGS versus less good implementation of contour gradient or LPFGS.

We first need to provide a function that evaluates the following two functions for a given input value $θ$: 

We can write a single function that returns both of these: 

```octave
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's **"fminunc()"** optimization algorithm along with the **"optimset()"** function that creates an object containing the options we want to send to "fminunc()". (Note: the value for MaxIter should be an integer, not a character string - errata in the video at 7:30) 

```octave
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function **"fminunc()"** our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

## Example 

![1st_example_of_costFunction_and_fminunc_in_octave](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/1st_example_of_costFunction_and_fminunc_in_octave.png)

Having implemented this cost function, you would, you can then call the advanced optimization function called the **'fminunc' - it stands for function minimization unconstrained in Octave** -and the way you call this is as follows. 

You set a few options. This is a options as a data structure that stores the options you want. *So **'GradObj' and 'on'** , these set the gradient objective parameter to on . It just means you are indeed going to provide a gradient to this algorithm ( which tells `fminunc` that our function returns both the cost and the gradient. This allows `fminunc` to use the gradient when minimizing the function ). I'm going to **set the maximum number of iterations to**, let's say, one hundred. We're going **give it an initial guess for $\theta$ **. There's a 2 by 1 vector. And then this command calls fminunc. This **'@' ** symbol presents a pointer to the cost function that we just defined up there. And if you call this, this will compute, you know, will use one of the more advanced optimization  algorithms.  And if you want to think it as just like gradient descent. But automatically choosing the learning rate alpha for so you don't have to do so yourself. But it will then attempt to use the sort of advanced optimization algorithms. Like gradient descent on steroids. To try to find the optimal value of theta for you.*  

 Let me actually show you what this looks like in Octave. So I've written this cost function of theta function exactly as we had it on the previous line. It computes J-val which is the cost function. And it computes the gradient with the two elements being the partial derivatives of the cost function with respect to, you know, the two parameters, theta one and theta two. 

```octave
function [jVal, gradient]=costFunction(theta)
　　jVal=(theta(1)-5)^2+(theta(2)-5)^2;
　　gradient=zeros(2,1);
　　gradient(1)=2*(theta(1)-5);
　　gradient(2)=2*(theta(2)-5);
end
```

Now let's switch to my Octave window. I'm gonna type in those commands I had just now. So, options equals optimset. 

```octave
options=optimset('GradObj','on','MaxIter',100);
initialTheta=zeros(2,1);
[optTheta, functionVal, exitFlag]=fminunc(@costFunction, initialTheta, options);
```

This is the notation for setting my parameters on my options, for my optimization algorithm. And if I hit enter this will run the optimization algorithm. And it returns pretty quickly. 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/10.png)

This funny formatting that's because my line, you know, my code wrapped around. So, this funny thing is just because my command line had wrapped around. But what this says is that numerically renders, you know, think of it as gradient descent on steroids, they found the optimal value of a theta is theta 1 equals 5, theta 2 equals 5, exactly as we're hoping for. **The `functionValue` at the optimum is essentially 10-to -the-minus-30-power. So that's essentially zero, which is also what we're hoping for** . And the `exitFlag` is 1, and this shows what the convergence status of this. And if you want you can do help `fminunc` to read the documentation for how to interpret the exit flag. But **the `exitFlag` let's you verify whether or not this algorithm thing has converged**. So that's how you run these algorithms in Octave. 

I should mention, by the way, that for the Octave implementation, this value of theta, your **parameter : vector of theta, its dimension is at least greater than or equal to 2.** So if theta is just a real number. So, if it is not at least a two-dimensional vector or some higher than two-dimensional vector, this `fminunc` may not work, so and if in case you have a one-dimensional function that you use to optimize, you can look in the octave documentation for `fminunc` for additional details. 

So, that's how we optimize our trial example of this simple quick driving cost function. However, we apply this to let's just say progression. In logistic regression we have a parameter vector theta, and I'm going to use a mix of octave notation and sort of math notation. But I hope this explanation will be clear, but our parameter vector theta comprises these parameters theta 0 through theta n because octave indexes, vectors using indexing from 1, you know, theta 0 is actually written theta 1 in octave, theta 1 is gonna be written. So, if theta 2 in octave and that's gonna be a written theta n+1, right? And that's because Octave indexes is vectors starting from index of 1 and so the index of 0. **So what we need to do then is write a cost function that captures the cost function for logistic regression. Concretely, the cost function needs to return `jVal`, which is, you know, `jVal` as you need some codes to compute J of theta and we also need to give it the gradient.** 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/11.png)

So, gradient 1 is going to be some code to compute the partial derivative in respect to theta 0, the next partial derivative respect to theta 1 and so on. Once again, this is gradient 1, gradient 2 and so on, rather than gradient 0, gradient 1 because octave indexes is vectors starting from one rather than from zero. But **the main concept I hope you take away from this slide is, that what you need to do, is write a function that returns the cost function and returns the gradient.** And so in order to apply this to logistic regression or even to linear regression, if you want to use these optimization algorithms for linear regression. What you need to do is plug in the appropriate code to compute these things over here. So, now you know how to use these advanced optimization algorithms.

**Because, using, because for these algorithms, you're using a sophisticated optimization library, it makes the just a little bit more opaque and so just maybe a little bit harder to debug. But because these algorithms often run much faster than gradient descent, often quite typically whenever I have a large machine learning problem, I will use these algorithms instead of using gradient descent. And with these ideas, hopefully, you'll be able to get logistic progression and also linear regression to work on much larger problems. So, that's it for advanced optimization concepts.** And in the next and final video on Logistic Regression, I want to tell you how to take the logistic regression algorithm that you already know about and make it work also on multi-class classification problems.

## Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}. 
$$
{% raw %}\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \\ & h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \\ & h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \\ & \cdots \\ & h_\theta^{(n)}(x) = P(y = n | x ; \theta) \\ & \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\\ \end{align*}{% endraw %}
$$
Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/06/12.png)

**To summarize:**   

1. Train a logistic regression classifier $h_θ(x)$ for each class￼ to predict the probability that ￼ ￼$y = i￼ $￼. 
2. To make a prediction on a new $x$ ,  pick the class ￼that maximizes $h_θ(x)$


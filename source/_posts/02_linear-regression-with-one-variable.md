---
title: 02_linear-regression-with-one-variable note2
date: 2018-01-02
copyright: true
categories: English
tags: [Machine Learning]
mathjax: true
mathjax2: true
---

## Note

This personal note is written after studying the coursera opening course, [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

## Model Representation

 To establish notation for future use, we’ll use $x^{(i)}$ to denote the **“input”variables** (living area in this example), also called **input features**, and $y^{(i)}$ to denote **the “output” or target variable** that we are trying to predict(price). A pair ( $x^{(i)},y^{(i)}$ ) is called a training example, and the dataset that we’ll be using to learn—a list of m training examples ( $x^{(i)},y^{(i)} ) ;i=1,...,m$ — is called a training set. **Note that the superscript “(i)” in the notation is simply an index into the training set**, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y to denote the space of output values. In this example, X = Y = ℝ. 

To describe the **supervised learning problem** slightly more formally, **our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y**. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this: 


![model_representation](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/model_representation.png)

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem.When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

## Cost Function

We can measure the accuracy of our hypothesis function by using a  **cost function** . This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's. 

$ J(θ_0,θ_1)={1\over2m}\sum\limits_{i=1}^m (\hat{y}_i−y_i)^2=\frac{1}{2m}\sum\limits_{i=1}^m(h_{θ(xi)}−y_i)^2$

To break it apart, it is ${1\over 2}\bar{x}$ where $\bar{x}$ is the mean of the squares of $h_{θ(xi)}−y_i$ , or the difference between the predicted value and the actual value. 

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $({1\over 2})$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $({1\over 2})$ term. The following image summarizes what the cost function does: 
![introduction_cost_function](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/introduction_cost_function.png)

##  Cost Function - Intuition I

 If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by $h_{θ(x)}$ ) which passes through these scattered data points. 

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(θ_0,θ_1)$ will be $0$. The following example shows the ideal situation where we have a cost function of $0$.

![cost_function_intuition_1](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_1.png)

​When $θ_1=1$, we get a slope of 1 which goes through every single data point in our model. Conversely, when $θ_1=0.5$, we see the vertical distance from our fit to the data points increase.                          

![cost_function_intuition_2](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_2.png)

This increases our cost function to $0.58​$. Plotting several other points yields to the following graph: 

![cost_function_intuition_3](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_3.png)

Thus as a goal, we should try to minimize the cost function. In this case, $θ_1=1$ is our global minimum.

## Cost Function - Intuition II

 A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_2-1.png)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(θ_0,θ_1)$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $θ_0 = 800$ and $θ_1= -0.15$ . Taking another $h(x)$ and plotting its contour plot, one gets the following graphs:

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_2-2.png)

When $θ_0 = 360$ and $θ_1 = 0$, the value of $J(θ_0,θ_1)$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_2-3.png)

The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around $0.12$ and $250$ respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

## Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in. 

Imagine that we graph our hypothesis function based on its fields $θ_0$ and $θ_1$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters. 

We put $θ_0$ on the x axis and $θ_1$ on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/gradient_descent.png)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.  The red arrows show the minimum points in the graph. 

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $α$ , which is called the learning rate. 

For example, the distance between each 'star' in the graph above represents a step determined by our parameter $α$ . A smaller $α$ would result in a smaller step and a larger $α$ results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(θ_0,θ_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places. 

The gradient descent algorithm is: 
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)
$$
repeat until convergence:

where $j=0,1$ represents the feature index number. 

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/Gradient_Descent_Algorithm.png)

At each iteration $j$ , one should simultaneously update the parameters $θ_1,θ_2,...,θ_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.

![ ](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/Gradient_Descent_code.png)

# Gradient Descent Intuition

In this video we explored the scenario where we used one parameter $θ_1$ and plotted its cost function to implement a gradient descent. Our formula for a single parameter was : 

Repeat until convergence:
$$
\theta_1:=\theta_1-\alpha \frac{d}{d\theta_1} J(\theta_1)
$$
Regardless of the slope's sign for $\frac{d}{d\theta_1} J(\theta_1)$, eventually converges to its minimum value. **The following graph shows that when the slope is negative, the value of $θ_1$ increases and when it is positive, the value of $θ_1$ decreases.**

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/cost_function_intuition_2-4.png)

On a side note, we should adjust our parameter $α$ to ensure that the gradient descent algorithm converges in a reasonable time. **Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.**

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/effect_of_too_large-or-small_gradient.png)

### How does gradient descent converge with a fixed step size α?

The intuition behind the convergence is that $\frac{d}{d\theta_1} J(\theta_1)$ , approaches 0 as we approach the bottom of our convex function. **At the minimum, the derivative will always be 0** and thus we get:
$$
\theta_1:=\theta_1-\alpha * 0
$$
![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/gradient_descent_at_a_fixed_learning_rate.png)
![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/gradient_descent_at_a_local_optima.png)

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/a_example_of_gradient_descent.gif)

 

# Gradient Descent For Linear Regression 

 

​     **Note:**    [At 6:15 " $h(x) = -900 - 0.1x$ " should be " $h(x) = 900 - 0.1x$ "]

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to : 
{% raw %}
$$
\begin{align*} \text{repeat until convergence: } \lbrace & \\ \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \\ \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \\ \rbrace& \end{align*}
$$
{% endraw %}
where m is the size of the training set, $θ_0$ a constant that will be changing simultaneously with $θ_1$ and $x_i,y_i$ are values of the given training set (data). 

Note that we have separated out the two cases for $θ_j$ into separate equations for $θ_0$ and $θ_1$ ;  and that for $θ_1$ we are multiplying $x_i$ at the end due to the derivative. The following is a derivation of $\frac{∂}{∂θ_j}J(θ)$ for a single example :
{% raw %}
$$
\begin{align*}
\frac{\partial}{\partial\theta_j}J(\theta) &=& \frac{\partial}{\partial\theta_j}\frac{1}{2}(h_{\theta}(x)-y)^2 \\
&=& 2 \cdot \frac{1}{2}(h_{\theta}(x)-y)\cdot \frac{\partial}{\partial\theta_j}(h_{\theta}(x)-y) \\
&=& (h_{\theta}(x)-y)\cdot\frac{\partial}{\partial\theta_j}\left(\sum\limits_{i=0}^{n}\theta_ix_i-y\right) \\
&=& (h_{\theta}(x)-y)x_j 
\end{align*}
$$
{% endraw %}

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate. 

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called  **batch gradient descent** . Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function.Here is an example of gradient descent as it is run to minimize a quadratic function.

![](http://pwmpcnhis.bkt.clouddn.com/snaildove.github.io/ml-andrew-ng/02/batch_gradient_descent.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at $(48,30)$. The $x$’s in the figure (joined by straight lines) mark the successive values of $θ$ that gradient descent went through as it converged to its minimum.


---
title: 02_logistic-regression-as-a-neural-network
date: 2018-02-02
copyright: true
categories: english
tags: [neural-networks-deep-learning, deep learning]
mathjax: true
mathjax2: true
---

## Note
This is my personal note at the 2nd week after studying the course [neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning/) and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

## 01_logistic-regression-as-a-neural-network

### 01_binary-classification

#### Binary Classification

In a binary classification problem, the result is a discrete value output. For example 
- account hacked (1) or compromised (0)
- a tumor malign (1) or benign (0)

**Example: Cat vs Non-Cat**
The goal is to train a classifier that the input is an image represented by a feature vector, $x$, and predicts whether the corresponding label $y$ is 1 or 0. In this case, whether this is a cat image (1) or a non-cat image (0).

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/1.png)

An image is store in the computer in three separate matrices corresponding to the Red, Green, and Blue color channels of the image. The three matrices have the same size as the image, for example, the resolution of the cat image is 64 pixels X 64 pixels, the three matrices (RGB) are 64 X 64 each.
The value in a cell represents the pixel intensity which will be used to create a feature vector of ndimension. In pattern recognition and machine learning, a feature vector represents an object, in this case, a cat or no cat. 
To create a feature vector, $x$, the pixel intensity values will be “unroll” or “reshape” for each color. The dimension of the input feature vector $x$ is $ n_x = 64 \times 64 \times 3 = 12 288.$
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/2.png)

#### notation

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/3.png)

### 02_Logistic Regression
Logistic regression is a learning algorithm used in a supervised learning problem when the output $y$ are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data.
#### Example: Cat vs No - cat
Given an image represented by a feature vector $x$, the algorithm will evaluate the probability of a cat being in that image.

$$\text{Civen }x, \hat{y}=P(y=1|x), \text{where } 0 \le \hat{y} \le 1$$

The parameters used in Logistic regression are:
• The input features vector: $x ∈ ℝ^{n_x}$, where $n_x$ is the number of features
• The training label: $y ∈ 0,1$
• The weights: $w ∈ ℝ^{n_x}$ , where $n_x$ is the number of features
• The threshold: $b ∈ ℝ$
• The output: $\hat{y} = \sigma(w^Tx+b)$
• Sigmoid function: $s = \sigma(w^Tx+b) = \sigma(z)= \frac{1}{1+e^{-z}}$

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/5.png)

$(w^Tx +b )$ is a linear function $(ax + b)$, but since we are looking for a probability constraint between [0,1], the sigmoid function is used. The function is bounded between [0,1] as shown in the graph above.
Some observations from the graph:
• If $z$ is a large positive number, then $\sigma(z) = 1$
• If $z$ is small or large negative number, then $\sigma(z) = 0$
• If $z$ = 0, then $\sigma(z) = 0.5$

#### notation

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/4.png)


### 03_logistic-regression-cost-function

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/6.png)

### 04_gradient-descent


![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/7.png)

### 05_06_derivatives

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/8.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/9.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/10.png)

### 07_computation-graph

You've heard me say that the computations of a neural network are organized in terms of a forward pass or a forward propagation step, in which we compute the output of the neural network, followed by a backward pass or back propagation step, which we use to compute gradients or compute derivatives. The computation graph explains why it is organized this way.
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/11.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/12.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/13.png)

### 09_logistic-regression-gradient-descent

Welcome back. In this video, we'll talk about how to compute derivatives for you to implement gradient descent for logistic regression. **The key takeaways will be what you need to implement. That is, the key equations you need in order to implement gradient descent for logistic regression**. In this video, I want to do this computation using the computation graph. I have to admit, using the computation graph is a little bit of an overkill for deriving gradient descent for logistic regression, but I want to start explaining things this way to get you familiar with these ideas so that, hopefully, it will make a bit more sense when we talk about fully-fledged neural networks. To that, let's dive into gradient descent for logistic regression. 

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/14.png)
In logistic regression, what we want to do is to modify the parameters, W and B, in order to reduce this loss.
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/15.png)

$$da = \frac{\partial{L}}{\partial{a}} =\frac{\partial \left\{ {-(ylog(a)+(1-y)log(1-a))} \right\} }{\partial{a}} = -\frac{y}{a} + \frac{1-y}{1-a} $$

$$dz=\frac{\partial{L}}{\partial{z}}=\frac{\partial{L}}{\partial{a}}\cdot \frac{\partial{a}}{\partial{z}} = \left(-\frac{y}{a} + \frac{1-y}{1-a}\right) \cdot a(1-a) = a - y$$

$$dw_1=\frac{\partial{L}}{\partial{w_1}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{w_1}} = x_1\cdot dz = x_1(a-y)$$

$$dw_2=\frac{\partial{L}}{\partial{w_2}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{w_2}} = x_2\cdot dz = x_2(a-y)$$

$$db=\frac{\partial{L}}{\partial{b}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{b}} = 1 \cdot dz = a - y$$

$$w_1 := w_1 - \alpha dw_1$$

$$w_2 := w_2 - \alpha dw_2$$

$$b := b - \alpha db$$

### 10_gradient-descent-on-m-examples

in a previous video you saw how to compute derivatives and implement gradient descent with respect to just one training example for religious regression now we want to do it for m training examples.

#### one single step gradient descent

![one single step gradient descent](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/16.png)

## 02_python-and-vectorization

### 01_vectorization

Welcome back. Vectorization is basically the art of getting rid of explicit folders in your code. In the deep learning era safety in deep learning in practice, you often find yourself training on relatively large data sets, because that's when deep learning algorithms tend to shine. And so, it's important that your code very quickly because otherwise, if it's running on a big data set, your code might take a long time to run then you just find yourself waiting a very long time to get the result. So in the deep learning era, I think the ability to perform vectorization has become a key skill. 

![what is vectorization](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/18.png)

![a example of the difference of run time between vectorization implementation and non-vectorization implementation](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/17.png)
Yeah. Vectorize version 1.5 milliseconds seconds and the four loop. So 481 milliseconds, again, about **300 times slower** to do the explicit four loop. If the engine x slows down, it's the difference between your code taking maybe one minute to run versus taking say five hours to run. And when you are implementing deep learning algorithms, you can really get a result back faster. It will be much faster if you vectorize your code. Some of you might have heard that a lot of **scaleable deep learning implementations** are done on a GPU or a graphics processing unit. But all the demos I did just now in the Jupiter notebook where actually on the CPU. And it turns out that both GPU and CPU have parallelization instructions. They're sometimes called **SIMD instructions**. This stands for a **single instruction multiple data**. But what this basically means is that, if you use built-in functions such as this np.function or other functions that don't require you explicitly implementing a for loop. It enables Phyton Pi to take much better advantage of parallelism to do your computations much faster. And this is true both computations on CPUs and computations on GPUs. It's just that GPUs are remarkably good at these SIMD calculations but CPU is actually also not too bad at that. Maybe just not as good as GPUs. You're seeing how vectorization can significantly speed up your code. **The rule of thumb to remember is whenever possible, avoid using explicit four loops.**

### 02_more-vectorization-examples
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/20.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/21.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/22.png)

### 03_vectorizing-logistic-regression

We have talked about how vectorization lets you speed up your code significantly. In this video, we'll talk about how you can vectorize the implementation of logistic regression, so they can process an entire training set, that is implement a single elevation of grading descent with respect to an entire training set without using even a single explicit for loop. I'm super excited about this technique, and when we talk about neural networks later without using even a single explicit for loop.


![vectorization implementation of logistic regression of forward of propagation](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/23.png)

Here are details of [python broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)


### 04_vectorizing-logistic-regressions-gradient-output

In the previous video, you saw how you can use vectorization to compute their predictions. The lowercase a's for an entire training set O at the same time. In this video, you see **how you can use vectorization to also perform the gradient computations for all M training samples**. Again, all sort of at the same time. And then at the end of this video, we'll put it all together and show how you can derive a very efficient implementation of logistic regression.


![vectorizing-logistic-regressions-gradient-output](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/24.png)

![vectorizing-logistic-regressions-gradient-output](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/25.png)


### 05_broadcasting-in-python


![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/26.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/29.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/27.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/28.png)

**Summary: Python or Numpy automatically expands two arrays or numbers to the same dimensions and operate element-wise.**

### 06_a-note-on-python-numpy-vectors

The ability of python to allow you to use broadcasting operations and more generally, **the great flexibility of the python numpy program language is, I think, both a strength as well as a weakness of the programming language**. I think it's a strength because they create expressivity of the language. A great flexibility of the language lets you get a lot done even with just a single line of code. But there's also weakness because with broadcasting and **this great amount of flexibility, sometimes it's possible you can introduce very subtle bugs or very strange looking bugs**, if you're not familiar with all of the intricacies of how broadcasting and how features like broadcasting work. **For example, if you take a column vector and add it to a row vector, you would expect it to throw up a dimension mismatch or type error or something. But you might actually get back a matrix as a sum of a row vector and a column vector. So there is an internal logic to these strange effects of Python.** But if you're not familiar with Python, I've seen some students have very strange, very hard to find bugs. So what I want to do in this video is share with you some couple tips and tricks that have been very useful for me to eliminate or simplify and eliminate all the strange looking bugs in my own code. **And I hope that with these tips and tricks, you'll also be able to much more easily write bug-free, python and numpy code**.

To illustrate one of the less intuitive effects of Python-Numpy, especially how you construct vectors in Python-Numpy, let me do a **quick demo**.

#### one rank array
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/30.png)

#### practical tips
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/31.png)

### 07_quick-tour-of-jupyter-ipython-notebooks

With everything you've learned, you're just about ready to tackle your first programming assignment. Before you do that, let me just give you a quick tour of iPython notebooks in Coursera.

Please see [the video]() to get details.

### 08_explanation-of-logistic-regression-cost-function-optional

![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/32.png)
![](http://pt8q6wt5q.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/33.png)

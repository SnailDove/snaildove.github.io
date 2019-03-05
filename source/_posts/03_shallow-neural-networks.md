---
title: 03_shallow-neural-networks
date: 2018-02-03
copyright: true
categories: english
tags: [neural-networks-deep-learning, deep learning]
mathjax: true
mathjax2: true
---

## Note
This is my personal note at the third week after studying the course [neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning/) and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

## 01_shallow-neural-network

### 01_neural-networks-overview

welcome back in this week's you learn to implement a neural network before diving into the technical details I wanted in this video to give you a quick overview of what you'll be seeing in this week's videos so if you don't follow all the details in this video don't worry about it we'll delve in the technical details in the next few videos but for now let's give a quick overview of how you implement in your network. 

![what's a Neural Network](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/1.png)

### 02_neural-network-representation

You see me draw a few pictures of neural networks. In this video, we'll talk about exactly what those pictures means. In other words, exactly what those neural networks that we've been drawing represent. And we'll start with focusing on the case of neural networks with what was called a single hidden layer. Here's a picture of a neural network. Let's give different parts of
these pictures some names.
![2 layer NN](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/2.png)

**note**:

1. The term hidden layer refers to the fact that in the training set, the true values for these nodes in the middle are not observed. That is you don't see what they should be in the training set. You see what the inputs are. You see what the output should be. But the things in the hidden layer are not seen in the training set. So that kind of explains the name hidden there just because you don't see it in the training set. 

2. One funny thing about notational conventions in neural networks is that this network that you've seen here is called a two layer neural network. And the reason is that when we count layers in neural networks, we don't count the input layer. So the hidden layer is layer one and the output layer is layer two. In our notational convention, we're calling the input layer layer zero, so technically maybe there are three layers in this neural network, because there's the input layer, the hidden layer, and the output layer. But in conventional usage, if you read research papers and elsewhere in the course, you see people refer to this particular neural network as a two layer neural network, because we don't count the input layer as an official layer.

3. the shape of the parameter $w$  between 2 layers is **(the_number_of_nodes_in_output_layer, the_number_of_nodes_in_intput_layer)**, that's merely a conventional way. And the parameter $b$ is only a constant, that's a (1, 1) matrix. 


### 03_computing-a-neural-networks-output

In the last video you saw what a single hidden layer neural network looks like in this video let's go through the details of exactly how this neural network computers outputs what you see is that is like logistic regression the repeat of all the times.


![2 layer NN](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/3.png)

![vectorization 2 layer NN](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/4.png)

![vectorization 2 layer NN](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/5.png)

### 04_vectorizing-across-multiple-examples

In the last video, you saw how to compute the prediction on a neural network, given a single training example. In this video, you see how to vectorize across multiple training examples. And the outcome will be quite similar to what you saw for logistic regression. Whereby stacking up different training examples in different columns of the matrix, you'd be able to take the equations you had from the previous video. And with very little modification, change them to make the neural network compute the outputs on all the examples on pretty much all at the same time. So let's see the details on how to do that. 

![vectorization 2 layer NN](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/6.png)

**note** : the row and column indices of the matrix $A, Z$ respectively correspond to the sequence numbers of the train examples and the nodes(units) in layers. And, the row and column indices of the matrix $X$ separately correspond to the sequence numbers of the train examples and the features of a example. Finally,  the row and column indices of the matrix $W$ separately correspond to the sequence numbers of the output units and the input units in the layer.

### 05_explanation-for-vectorized-implementation

In the previous video, we saw how with your training examples stacked up horizontally in the matrix $X$, you can derive a vectorized implementation for propagation through your neural network. Let's give a bit more justification for why the equations we wrote down is a correct implementation of vectorizing across multiple examples.

So let's go through part of the propagation calculation for the few examples.
![](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/7.png)
![](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/8.png)

### 06_activation-functions

When you boost a neural network, one of the choices you get to make is what activation functions use independent layers as well as at the output unit of your neural network so far we've just been using the sigmoid activation function but sometimes other choices can work much better let's take a look at some of the options.

![](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/9.png)

In the forward propagation steps for the neural network we have these two steps where we use the sigmoid function here so that sigmoid is called an **activation function**, so in the more general case we can have a different function G of z, $g(z)$, where G could be a nonlinear function that may not be the sigmoid function.       

So for example the sigmoid function goes between 0 & 1, an activation function that almost always works better than the sigmoid function is the tanh function or the hyperbolic tangent function. The **tanh** function or the **hyperbolic tangent** function is actually mathematically a shifted version of the sigmoid function so as a you know sigmoid function just like that but shift it so that it now crosses a zero zero point and rescale. So it goes to minus one and plus one and it turns out that for hidden units if you let the function G of Z, $g(z)$, be equal to $tanh(z)$. This almost always works better than the sigmoid function because with values between plus one and minus one. The mean of the activations that come out of your head in there are closer to having a zero mean and so just as sometimes when you train a learning algorithm you might Center the data and have your data have zero mean. Using a $tanh$ function instead of a $sigmoid$ function kind of has the effect of centering your data, so that the mean of the data is close to the zero rather than maybe a 0.5 and this actually makes learning for the next layer a little bit easier we'll say more about this in the second course when we talk about optimization algorithms as well but **one takeaway is that I pretty much never use the sigmoid activation function anymore, the tanh function is almost always strictly superior**. **The only one exception** is for the output layer because if Y is either 0 or 1 then it makes sense for $\hat{y}$ to be a number that you want to output. There's between 0 and 1 rather than between minus 1 and 1 so the one exception where I would use the sigmoid activation function is when you're using binary classification in which case you might use the sigmoid activation function for the output layer.


Now **one of the downsides of both the sigmoid function and the tanh function** is that if Z is either very large or very small then the gradient of the derivative or the slope of this function becomes very small so Z is very large or Z is very small the slope of the function you know ends up being close to zero and so this can slow down gradient descent, so one of the toys that is very popular in machine learning is what's called the **rectified linear unit, ReLU**. so the value function looks like the down-left function of the above slide and the formula is a equals max of 0 comma Z, $max=\{0, z\}$ , So the derivative is 1 so long as Z is positive and derivative or the slope is 0 when Z is negative. If you're implementing this technically the derivative when Z is exactly 0 is not well-defined but when you implement is in the computer, the odds that you get exactly the equals 0 0 0 0 0 0 0 0 0 0 it is very small, so you don't need to worry about it. In practice you could pretend a derivative when Z is equal to 0 you can pretend is either 1 or 0 and you can work just fine athlough the fact that is not differentiable.

So here are **some rules of thumb for choosing activation functions**: 

If your output is 0 1 value if you're I'm using binary classification then the sigmoid activation function is very natural for the output layer and then for all other units on ReLU or the rectified linear unit is increasingly the default choice of activation function. so if you're not sure what to use for your head in there I would just use the relu activation function that's what you see most people using these days although sometimes people also use the tannish activation function. **One advantage of the ReLU** is that the derivative is equal to zero when z is negative in practice this works just fine, but there is another version of the ReLU called the **leaky ReLU** will give you the formula on the next slide, but instead of it being zero** when z is negative it just takes a slight slope** like the down-right of the above slide. So this is called the leaky ReLU. This usually works better than the RELU activation function although it's just not used as much in practice. Either one should be fine although if you had to pick one I usually just use the revenue and **the advantage of both the ReLU and the leaky ReLU** is that for a lot of the space of Z the derivative of the activation function the slope of the activation function is very different from zero and so in practice using the ReLU activation function your new network will often learn much faster than using the tanh or the sigmoid activation function and the main reason is that on this less of this effect of the slope of the function going to zero which slows down learning and I know that for half of the range of Z the slope of relu is zero but in practice enough of your hidden units will have Z greater than zero so learning can still be quite mask for most training examples.

On the above slide, the leaky ReLU is $max=\{0.01z, z\}$. You might say you know why is that constant 0.01 well you can also make that another parameter of the learning algorithm and some people say that works even better but I hardly see people do that so but if you feel like trying it in your application you know please feel free to do so and and you can just see how it works and how long works and stick with it if it gives you good result.

One of the themes we'll see in deep learning is that you often have a lot of different choices in how you code your neural network ranging from number of credit units to the chosen activation function to how you initialize the parameters which we'll see later a lot of choices like that and it turns out that is sometimes difficult to get good guidelines for exactly what will work best for your problem so so these three courses. I'll keep on giving you a sense of what I see in the industry in terms of what's more or less popular but for your application with your applications idiosyncrasy. **It's actually very difficult to know in advance exactly what will work best so a piece of advice would be if you're not sure which one of these activation functions work best you know try them all and then evaluate on like a holdout validation set or like a development set which we'll talk about later and see which one works better and then go of that** and I think that by testing these different choices for your application you'd be better at future proofing your neural network architecture against the idiosyncrasy in our problem as well evolutions of the algorithms rather than you know if I were to tell you always use a ReLU activation and don't use anything else that just may or may not apply for whatever problem you end up working on you know either either in the near future.

### 07_why-do-you-need-non-linear-activation-functions

Why does your nerual network need a nonlinear activation function turns out that for your new network to compute interesting functions you do need to take a nonlinear activation function unless you want.

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/10.png)

No matter how many layers your neural network has always doing is just computing a linear activation function so you might as well not have any hidden layers some of the cases that briefly mentioned it turns out that if you have a linear activation function here and a sigmoid function here(on output layer) then this model is no more expressive than standard logistic regression without any hidden layer so I won't bother to prove that but you could try to do so if you want but the take-home is that a linear hidden layer is more or less useless because **the composition of two linear functions is itself a linear function**.


there is **just one place where you might use a linear activation function G of Z equals Z and that's if you are doing machine learning on a regression problem so if Y is a real number** so for example if you're trying to predict housing prices so Y is a it's not 0 1 but it's a real number you know anywhere from zero dollars is a price of holes up to however expensive right house of kin I guess maybe however can be you know potentially millions of dollars so however however much houses cost in your data set but if Y takes on these real values then it might be OK to have a linear activation function here so that your output Y hat is also a real number going anywhere from minus infinity to plus infinity but then the hidden units should not use the new activation functions they could use relu or 10h or these you relu or maybe something else so the one place you might use a linear activation function others usually in the output layer but other than that using a linear activation function in a fitting layer except for some very special circumstances relating to compression that won't want to talk about using a linear activation function is extremely rare oh and of course today actually predicting housing prices as you saw on the week 1 video because housing prices are all non-negative perhaps even then you can use a value activation function so that your outputs Y hat are all greater than or equal to 0.

so I hope that gives you a sense of why having a nonlinear activation function is a critical part of neural networks.

### 08_derivatives-of-activation-functions

When you implement back-propagation for your neural network you need to really compute the slope or the derivative of the activation functions so let's take a look at our choices of activation functions and how you can compute the slope of these functions.

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/11.png)

Sometimes instead of writing this thing $\frac{dg(z)}{dz}$, the shorthand for the derivative is G prime of Z, $g'(z)$ . so G prime of Z in calculus the the little dash on top is called **prime** but so G prime of Z is a shorthand for the in calculus for the derivative of the function of G with respect to the input variable Z and then in a new network we have $a = g(z)$,  right equals this then this formula also simplifies to $a(1-a)$. 

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/12.png)

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/13.png)

### 09_gradient-descent-for-neural-networks

All right I think that's be an exciting video in this video you see how to implement gradient descent for your neural network with one hidden layer in this video I'm going to just give you the equations you need to implement in order to get back propagation of the gradient descent working and then in the video after this one I'll give some more intuition about why these particular equations are the accurate equations or the correct equations for computing the gradients you need for your neural network.


In logistic regression, what we want to do is to modify the parameters, W and B, in order to reduce this loss.
![](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/02_neural-networks-basics/15.png)

$$da = \frac{\partial{L}}{\partial{a}} =\frac{\partial \left\{ {-(ylog(a)+(1-y)log(1-a))} \right\} }{\partial{a}} = -\frac{y}{a} + \frac{1-y}{1-a} $$

$$dz=\frac{\partial{L}}{\partial{z}}=\frac{\partial{L}}{\partial{a}}\cdot \frac{\partial{a}}{\partial{z}} = \left(-\frac{y}{a} + \frac{1-y}{1-a}\right) \cdot a(1-a) = a - y$$

$$dw_1=\frac{\partial{L}}{\partial{w_1}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{w_1}} = x_1\cdot dz = x_1(a-y)$$

$$dw_2=\frac{\partial{L}}{\partial{w_2}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{w_2}} = x_2\cdot dz = x_2(a-y)$$

$$db=\frac{\partial{L}}{\partial{b}}=\frac{\partial{L}}{\partial{z}}\cdot \frac{\partial{z}}{\partial{b}} = 1 \cdot dz = a - y$$

$$w_1 := w_1 - \alpha dw_1$$

$$w_2 := w_2 - \alpha dw_2$$

$$b := b - \alpha db$$

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/14.png)

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/15.png)


### 11_random-initialization

When you change your neural network, it's important to initialize the weights randomly. For logistic regression, it was okay to initialize the weights to zero. But for a neural network of initialize the weights to parameters to all zero and then applied gradient descent, it won't work. Let's see why.

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/16.png)


So you have here two input features, so n0=2, and two hidden units, so n1=2. And so the matrix associated with the hidden layer, w 1, is going to be two-by-two. Let's say that you initialize it to all 0s, so 0 0 0 0, two-by-two matrix. And let's say B1 is also equal to 0 0. It turns out initializing the bias terms b to 0 is actually okay, but initializing w to all 0s is a problem. So the problem with this formalization is that for any example you give it, you'll have that a1,1 and a1,2, will be equal, right? So this activation and this activation will be the same, because both of these hidden units are computing exactly the same function. And then, when you compute backpropagation, it turns out that dz11 and dz12 will also be the same colored by symmetry, right? Both of these hidden unit will initialize the same way. Technically, for what I'm saying, I'm assuming that the outgoing weights or also identical. So that's w2 is equal to 0 0. But if you initialize the neural network this way, then this hidden unit and this hidden unit are completely identical. Sometimes you say they're completely symmetric, which just means that they're completing exactly the same function. And by kind of a proof by induction, it turns out that after every single iteration of training your two hidden units are still computing exactly the same function. Since [INAUDIBLE] show that dw will be a matrix that looks like this. Where every row takes on the same value. So we perform a weight update. So when you perform a weight update, w1 gets updated as w1- alpha times dw. You find that w1, after every iteration, will have the first row equal to the second row. So it's possible to construct a proof by induction that if you initialize all the ways, all the values of w to 0, then because both hidden units start off computing the same function. And both hidden the units have the same influence on the output unit, then after one iteration, that same statement is still true, the two hidden units are still symmetric. And therefore, by induction, after two iterations, three iterations and so on, no matter how long you train your neural network, both hidden units are still computing exactly the same function. And so in this case, there's really no point to having more than one hidden unit. Because they are all computing the same thing. And of course, for larger neural networks, let's say of three features and maybe a very large number of hidden units, a similar argument works to show that with a neural network like this. [INAUDIBLE] drawing all the edges, if you initialize the weights to zero, then all of your hidden units are symmetric. And no matter how long you're upgrading the center, all continue to compute exactly the same function. So that's not helpful, because you want the different hidden units to compute different functions. 

The solution to this is to initialize your parameters randomly. So here's what you do.

![why-do-you-need-non-linear-activation-functions](http://pne0wr4lu.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/17.png)

You can set w1 = np.random.randn. This generates a gaussian random variable (2,2). And then usually, you multiply this by very small number, such as 0.01. So you initialize it to very small random values. And then b, **it turns out that b does not have the symmetry problem, what's called the symmetry breaking problem**. So it's okay to initialize b to just zeros. Because so long as w is initialized randomly, you start off with the different hidden units computing different things. And so you no longer have this symmetry breaking problem. And then similarly, for w2, you're going to initialize that randomly. And b2, you can initialize that to 0. **So you might be wondering, where did the constant come from and why is it 0.01? Why not put the number 100 or 1000? Turns out that we usually prefer to initialize the weights to very small random values. Because if you are using a tanh or sigmoid activation function, or the other sigmoid, even just at the output layer. If the weights are too large, then when you compute the activation values, remember that z[1]=w1 x + b. And then a1 is the activation function applied to z1. So if w is very big, z will be very, or at least some values of z will be either very large or very small. And so in that case, you're more likely to end up at these fat parts of the tanh function or the sigmoid function, where the slope or the gradient is very small. Meaning that gradient descent will be very slow. So learning was very slow. So just a recap, if w is too large, you're more likely to end up even at the very start of training, with very large values of z. Which causes your tanh or your sigmoid activation function to be saturated, thus slowing down learning. If you don't have any sigmoid or tanh activation functions throughout your neural network, this is less of an issue. But if you're doing binary classification, and your output unit is a sigmoid function, then you just don't want the initial parameters to be too large. So that's why multiplying by 0.01 would be something reasonable to try, or any other small number**. And same for w2, right? This can be random.random. I guess this would be 1 by 2 in this example, times 0.01. Missing an s there. So finally, it turns out that sometimes they can be better constants than 0.01. When you're training a neural network with just one hidden layer, it is a relatively shallow neural network, without too many hidden layers. Set it to 0.01 will probably work okay. But when you're training a very very deep neural network, then you might want to pick a different constant than 0.01. And in next week's material, we'll talk a little bit about how and when you might want to choose a different constant than 0.01. But either way, it will usually end up being a relatively small number. So that's it for this week's videos. You now know how to set up a neural network of a hidden layer, initialize the parameters, make predictions using. As well as compute derivatives andimplement gradient descent, using backprop. 

So that,you should be able to do the quizzes, as well as this week's programming exercises. Best of luck with that. I hope you have fun with the problem exercise, and look forward to seeing you in the week four materials.



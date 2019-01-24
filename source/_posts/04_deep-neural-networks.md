---
title: 04_deep-neural-network
date: 2018-02-04
copyright: true
categories: english
tags: [neural-networks-deep-learning, deep learning]
mathjax: true
mathjax2: true
---

## Note
This is my personal note at the 4th week after studying the course [neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning/) and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).


## 01_deep-neural-network

Welcome to the fourth week of this course. By now, you've seen four promulgation and back promulgation in the context of a neural network, with a single hidden layer, as well as logistic regression, and you've learned about vectorization, and when it's important to initialize the ways randomly. If you've done the past couple weeks homework, you've also implemented and seen some of these ideas work for yourself. So by now, you've actually seen most of the ideas you need to implement a deep neural network. What we're going to do this week, is take those ideas and put them together so that you'll be able to implement your own deep neural network. Because this week's problem exercise is longer, it just has been more work, I'm going to keep the videos for this week shorter as you can get through the videos a little bit more quickly, and then have more time to do a significant problem exercise at then end, which I hope will leave you having thoughts deep in neural network, that if you feel proud of.

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/1.png)

but over the last several years the AI, on the machine learning community, has realized that there are functions that very deep neural networks can learn that shallower models are often unable to. Although for any given problem, it might be hard to predict in advance exactly how deep in your network you would want. So it would be reasonable to try logistic regression, try one and then two hidden layers, and view the number of hidden layers as another hyper parameter that you could try a variety of values of, and evaluate on all that across validation data, or on your development set. See more about that later as well.

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/2.png)


## 02_forward-propagation-in-a-deep-network

In the last video we distract what is the deep neural network and also talked about the notation we use to describe such networks in this video you see how you can perform for propagation in a deep network.

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/3.png)

One of the ways to increase your odds of having bug-free implementation is to think very systematic and carefully about the matrix dimensions you're working with so when I'm trying to develop my own code I often pull a piece of paper and just think carefully through so the dimensions of the matrix I'm working with let's see how you could do that in the next video.


## 03_getting-your-matrix-dimensions-right

When implementing a deep neural network, one of the debugging tools I often use to check the correctness of my code is to pull a piece of paper, and just work through the dimensions and matrix I'm working with. So let me show you how to do that, since I hope this will make it easier for you to implement your deep nets as well.

### one training example
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/4.png)


$\because \text{the dimensions of x}(a^{[0]}) \text{: } (n^{[0]}, 1)$

$\therefore $
$W^{[l]}: (n^{[l]}, n^{[l-1]})$

$b^{[l]}: (n^{[l]}, 1)$

$dW^{[l]}: (n^{[l]}, n^{[l-1]})$

$db^{[l]}: (n^{[l]}, 1)$

$dz^{[l]}: (n^{[l]}, 1)$

$da^{[l]}: (n^{[l]}, 1)$ is the same shape of $z^{[l]}$.


### m training examples
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/5.png)

$\because \text{the dimensions of X}(A^{[0]}) \text{: } (n^{[0]}, m)$

$\therefore$

$W^{[l]} : (n^{[l]}, n^{[l-1]})$

$B^{[l]} : (n^{[l]}, m) $

$dW^{[l]} : (n^{[l]}, n^{[l-1]})$

$dB^{[l]} : (n^{[l]}, m)$

$dZ^{[l]} : (n^{[l]}, m)$

$dA^{[l]} : (n^{[l]}, m) \text{ is the same shape of } Z^{[l]}$


So I hope the little exercise we went through helps clarify the dimensions that the various matrices you'd be working with. **When you implement backpropagation for a deep neural network, so long as you work through your code and make sure that all the matrices' dimensions are consistent. That will usually help, it'll go some ways toward eliminating some cause of possible bugs**. So I hope that exercise for figuring out the dimensions of various matrices you'll been working with is helpful. When you implement a deep neural network, if you keep straight the dimensions of these various matrices and vectors you're working with. Hopefully they'll help you eliminate some cause of possible bugs, **it certainly helps me get my code right**. So next, we've now seen some of the mechanics of how to do forward propagation in a neural network. But why are deep neural networks so effective, and why do they do better than shallow representations? Let's spend a few minutes in the next video to discuss that.

## 04_why-deep-representations

We've all been hearing that deep neural networks work really well for a lot of problems, and it's not just that they need to be big neural networks, is that specifically, they need to be deep or to have a lot of hidden layers. So why is that? Let's go through a couple examples and try to gain some intuition for why deep networks might work well.

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/6.png)

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/7.png)


To compute $y=x_{1}\oplus x_{2}\oplus x_{3}\oplus \cdots\oplus x_{n}$, the depth of deep neural network is $O(log_{2}^{n})$, the number of activiation units or nodes is $2^{\log_{2}(n)-1} + \cdots + 2 + 1 = 1\cdot \dfrac{1-2^{\log_{2}(n)}}{1-2}=2^{\log_{2}(n)}-1=n-1$. But in one-hidden-layer neural network, the number of activiation units or nodes is $2^{n-1}$.


Now, in addition to this reasons for preferring deep neural networks to be roughly on, is I think the other reasons the term deep learning has taken off is just branding. This things just we call neural networks belong to hidden layers, but the phrase deep learning is just a great brand, it's just so deep. So I think that once that term caught on that really new networks rebranded or new networks with many hidden layers rebranded, help to capture the popular imagination as well. They regard as the PR(public relations) branding deep networks do work well. Sometimes people go overboard and insist on using tons of hidden layers. **But when I'm starting out a new problem, I'll often really start out with even logistic regression then try something with one or two hidden layers and use that as a hyper parameter. Use that as a parameter or hyper parameter that you tune in order to try to find the right depth for your neural network. But over the last several years there has been a trend toward people finding that for some applications, very, very deep neural networks here with maybe many dozens of layers sometimes, can sometimes be the best model for a problem**. So that's it for the intuitions for why deep learning seems to work well. Let's now take a look at the mechanics of how to implement not just front propagation, but also back propagation.

### 05_building-blocks-of-deep-neural-networks

In the earlier videos from this week as well as from the videos from the past several weeks you've already seen the basic building blocks of board propagation and back propagation the key components you need to implement a deep neural network let's see how you can put these components together to build a deep net use the network with a few layers.
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/8.png)
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/9.png)

### 06_forward-and-backward-propagation

In a previous video you saw the basic blocks of implementing a deep neural network for propagation step for each layer and a corresponding backward propagation step let's see how you can actually implement these steps.

![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/10.png)
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/11.png)
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/12.png)

although I have to say you know even today when I implement a learning algorithm sometimes even I'm surprised when my learning algorithm implementation works and it's because longer complexity of machine learning comes from the data rather than from the lines of code so sometimes you feel like you implement a few lines of code not question what it did but there's almost magically work and it's because of all the magic is actually not in the piece of code you write which is often you know not too long it's not it's not exactly simple but there's not you know 10,000 100,000 lines of code but you feed it so much data that sometimes even though I work the machine only for a long time sometimes it's so you know surprises me a bit when my learning algorithm works because lots of complexity of your learning algorithm comes from the data rather than necessarily from your writing you know thousands and thousands of lines of code all right so that's um how do you implement deep neural networks and again this will become more concrete when you've done the programming exercise before moving on I want to discuss in the next video want to discuss hyper parameters and parameters it turns out that when you're training deep nets being able to organize your hyper params as well will help you be more efficient in developing your networks in the next video let's talk about exactly what that means.

### 07_parameters-vs-hyperparameters

Being effective in developing your deep neural Nets requires that you not only organize your parameters well but also your hyper parameters, so what are hyper parameters.


![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/13.png)

so **when you're training a deep net for your own application you find that there may be a lot of possible settings for the hyper parameters that you need to just try out so applied deep learning today is a very empirical process where often you might have an idea**. 
![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/14.png)
For example you might have an idea for the best value for the learning rate you might say well maybe alpha equals 0.01 I want to try that then you implemented try it out and then see how that works and then based on that outcome you might say you know what I've changed online I want to increase the learning rate to 0.05 and so if you're not sure what's the best value for the learning ready-to-use you might try one value of the learning rate alpha and see their cost function j go down like this then you might try a larger value for the learning rate alpha and see the cost function blow up and diverge then you might try another version and see it go down really fast it's inverse to higher value you might try another version and see it you know see the cost function J do that then I'll be China so the values you might say okay looks like this the value of alpha gives me a pretty fast learning and allows me to converge to a lower cost function jennice I'm going to use this value of alpha.

you saw in a previous slide that there are a lot of different hybrid parameters and **it turns out that when you're starting on the new application I should find it very difficult to know in advance exactly what's the best value of the hyper parameters so what often happen is you just have to try out many different values and go around this cycle your trial some value really try five hidden layers with this many number of hidden units implement that see if it works and then iterate so the title of this slide is that apply deep learning is very empirical process** and empirical process is maybe a fancy way of saying you just have to try a lot of things and see what works.

another effect I've seen is that deep learning today is applied to so many problems ranging from computer vision to speech recognition to natural language processing to a lot of structured data applications such as maybe a online advertising or web search or product recommendations and so on and what I've seen is that **first** I've seen researchers from one discipline any one of these try to go to a different one and sometimes the intuitions about hyper parameters carries over and sometimes it doesn't so I often advise people especially when starting on a new problem to just try out a range of values and see what works and then the next course we'll see a systematic way for trying out a range of values all right and **second** even if you're working on one application for a long time, you know, maybe you're working on online advertising. As you make progress on the problem, it is quite possible there the best value for the learning rate, a number of hidden units and so on might change, so even if you tune your system to the best value of hyper parameters to daily as possible you find that the best value might change a year from now. Maybe because the computer infrastructure I'd be you know CPUs or the type of GPU running on or something has changed, but so **maybe one rule of thumb is you know every now and then maybe every few months if you're working on a problem for an extended period of time for many years just try a few values for the hyper parameters and double check if there's a better value for the hyper parameters and as you do**. So you slowly gain intuition as well about the hyper parameters that work best for your problems and I know that this might seem like an unsatisfying part of deep learning that you just have to try on all the values for these hyper parameters but maybe this is one area where deep learning research is still advancing and maybe over time we'll be able to give better guidance for the best hyper parameters to use, but it's also possible that because CPUs and GPUs and networks and data says are all changing and it is possible that the guidance won't to converge for some time and you just need to keep trying out different values and evaluate them on a hold on cross-validation set or something and pick the value that works for your problems. So that was a brief discussion of hyper parameters in the second course we'll also give some suggestions for how to systematically explore the space of hyper parameters but by now you actually have pretty much all the tools you need to do their programming exercise before you do that adjust or share view one more set of ideas which is I often ask what does deep learning have to do the human brain.

### 08_what-does-this-have-to-do-with-the-brain 

So what a deep learning have to do the punchline I would say not a whole lot but let's take a quick look at why people keep making the analogy between deep learning and the human brain.


![](http://pltr89sz6.bkt.clouddn.com/gitpage/deeplearning.ai/neural-networks-deep-learning/03_shallow-neural-networks/04_deep-neural-networks/15.png)

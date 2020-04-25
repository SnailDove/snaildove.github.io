---
title: 08_neural-networks-representation note8
date: 2018-01-08
copyright: true
categories: English
tags: [Machine Learning]
mathjax: true
mathjax2: true
---

# Note

This personal note is written after studying the opening course on [the coursera website](https://www.coursera.org), [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

## Motivations

### non-linear-hypotheses

In order to motivate the discussion of neural networks, let me start by showing you a few examples of machine learning problems  where we need to learn complex non-linear hypotheses. Consider a supervised learning classification problem where you have a training set like this. If you want to apply logistic regression to this problem, one thing you could do is apply logistic regression with a lot of nonlinear features like that. So here, g as usual is the sigmoid function, and we can include lots of polynomial terms like these.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/1.png)

And, if you include enough polynomial terms then,  you know, maybe you can get a hypotheses that separates the positive and negative examples.This particular method works well when you have only, say,  two features - x1 and x2 because you can then include all those polynomial terms of x1 and x2. 

#### House Prediction

But for many interesting machine learning problems would have a lot more features than just two. We've been talking for a while about housing prediction, and suppose you have a housing classification problem rather than a regression problem, like maybe if you have different features of a house, and you want
to predict what are the odds that your house will be sold within the next six months, so that will be a classification problem. And as we saw we can come up with quite a lot of features, maybe a hundred different features of different houses. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/2.png)

For a problem like this, if you were to include all the quadratic terms, all of these, even all of the quadratic that is the second or the polynomial terms, there would be a lot of them. There would be terms like x1 squared, x1x2, x1x3, you know, x1x4 up to x1x100 and then you have x2 squared, x2x3 and so on. And if you include just the second order terms, that is, the terms that are a product of, you know,  two of these terms, x1 times x1 and so on, then, for the case of n equals 100, you end up with about five thousand features. And, asymptotically, the number of quadratic features grows roughly as order n squared, where n is the number of the original features, like x1 through x100 that we had. And its actually closer to n squared over two.  So including all the quadratic features doesn't seem like it's maybe a good idea, because that is a lot of features and you might up overfitting the training set, and it can also be computationally expensive, you know, to be working with that many features. One thing you could do is include only a subset of these, so if you include only the features x1 squared, x2 squared, x3 squared, up to maybe x100 squared, then the number of features is much smaller. Here you have only 100 such  quadratic features, but this is not enough features and certainly won't let you fit the data set like that on the upper left. In fact, if you include only these quadratic features together with the original x1, and so on, up to x100 features, then you can actually fit very interesting hypotheses. So, you can fit things like, you know, access a line of the ellipses like these, but you certainly cannot fit a more complex data set like that shown here. So 5000 features seems like a lot, if you were to  include the cubic, or third order known of each others, the x1, x2, x3. You know, x1 squared, x2, x10 and x11, x17 and so on. You can imagine there are gonna be a lot of these features.

In fact, they are going to be order and cube such features and if any is 100 you can compute that, you end up with on the order of about 170,000 such cubic features and so including these higher auto-polynomial features when your original feature set end is large this really dramatically blows up your feature space and this doesn't seem like a good way to come up with additional features with which to build none many classifiers when n is large. 

#### Car recognition

**For many machine learning problems, n will be pretty large**.  Here's an example. Let's consider the problem of computer vision. And suppose you want to use machine learning to train a classifier to examine an image and tell us whether or not the image is a car. Many people wonder why computer vision could be difficult. I mean when you and I look at this picture it is so obvious what this is. You wonder how is it that a learning algorithm could possibly fail to know what this picture is.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/3.png)

To understand why computer vision is hard let's zoom into a small part of the image like that area where the little red rectangle is. It turns out that where you and I see a car, the computer sees that. What it sees is this matrix, or this grid, of pixel intensity values that tells us the brightness of each pixel in the image.So the computer vision problem is to look at this matrix of pixel intensity values, and tell us that these numbers represent the door handle of a car. Concretely, when we use machine learning to build a car detector, what we do is we come up with a label training set, with, let's say, a few label examples of cars and a few label examples of things that are not cars, then we give our training set to the learning algorithm trained a classifier and then, you know, we may test it and show the new image and ask, "What is this new thing?". And hopefully it will recognize that that is a car.

To understand why we need nonlinear hypotheses, let's take a look at some of the images of cars and maybe non-cars that we might feed to our learning algorithm. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/4.png)

Let's pick a couple of pixel locations in our images, so that's pixel one location and pixel two location, and let's plot this car, you know, at the location, at a certain point, depending on the intensities of pixel one and pixel two. And let's do this with a few other images. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/5.png)

So let's take a different example of  the car and you know, look at the same two pixel locations and that image has a different intensity for pixel one and a different intensity for pixel two. So, it ends up at a different location on the figure. And then let's plot some negative examples as well. That's a non-car, that's a non-car. And if we do this for more and more examples using the pluses(+) to denote cars and minuses(-) to denote non-cars, what we'll find is that the cars and non-cars end up lying in different regions of the space, and what we need therefore is some sort of non-linear hypotheses to try to separate out the two classes. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/6.png)

What is the dimension of the feature space? Suppose we were to use just 50 by 50 pixel images. Now that suppose our images were pretty small ones, just 50 pixels on the side. Then we would have 2500 pixels, and so the dimension of our feature size will be N equals 2500 where our feature vector x is a list of all the pixel testings, you know,  the pixel brightness of pixel one, the brightness of pixel two, and so on down to the pixel brightness of the last pixel where, you know, in a typical  computer representation, each of these may be values between say 0 to 255 if it gives us the grayscale value. So we have n equals 2500, and that's if we were using grayscale images. If we were using RGB images with separate red, green and blue values, we would have n equals 7500. So, if we were to try to learn a nonlinear hypothesis by including all the quadratic features, that is all the terms of the form, you know, $X_i$ times $X_j$, while with the 2500 pixels we would end up with a total of three million features. And that's just too large to be reasonable; the computation would be very expensive to find and to represent all of these three million features per training example. 

**So, simple logistic regression together with adding in maybe the quadratic or the cubic features that's just not a good way to learn complex nonlinear hypotheses when n is large because you just end up with too many features. **

In the next few videos, I would like to tell you about **Neural Networks, which turns out to be a much better way to learn complex hypotheses, complex nonlinear hypotheses even when your input feature space, even when n is large.** And along the way I'll also get to show you a couple of fun videos of historically important applications of Neural networks as well that I hope those videos that we'll see later will be fun for you to watch as well.

### neurons and the brain

Neural Networks are a pretty old algorithm that was originally motivated by the goal of having machines that can mimic the brain. 

Now in this class, of course I'm teaching Neural Networks to you because they work really well for different machine learning problems and not, certainly not because they're logically motivated. In this video, I'd like to give you some of the background on Neural Networks. So that we can get a sense of what we can expect them to do. Both in the sense of applying them to modern day machinery problems, as well as for those of you that might be interested in maybe the big AI dream of someday building truly intelligent machines. Also, how Neural Networks might pertain to that. **The origins of Neural Networks was as algorithms that try to mimic the brain** and those a sense that if we want to build learning systems while why not mimic perhaps the most amazing learning machine we know about, which is perhaps the brain. 

Neural Networks came to be very widely used throughout the 1980's and 1990's and for various reasons as popularity diminished in the late 90's. But more recently, Neural Networks  have had a major recent resurgence. One of the reasons for this resurgence is that Neural Networks are computationally some what more expensive algorithm and so, it was only, you know, maybe somewhat more recently that computers became fast enough to really run large scale Neural Networks and because of that as well as a few other technical reasons which we'll talk about later, modern Neural Networks today are the state of the art technique for many applications.

So, when you think about mimicking the brain while one of the human brain does tell me same things, right? The brain can learn to see process images than to hear, learn to process our sense of touch. We can, you know, learn to do math, learn to do calculus, and the brain does so many different and amazing things. It seems like if you want to mimic the brain it seems like you have to write lots of different pieces of software to mimic all of these different fascinating, amazing things that the brain tell us, but does is this fascinating hypothesis that the way the brain does all of these different things is not worth like a thousand different programs, but instead, the way the brain does it is worth just a single learning algorithm. This is just a hypothesis but let me share with you some of the evidence for this.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/7.png)

This part of the brain, that little red part of the brain, is your auditory cortex and the way you're understanding my voice now is your ear is taking the sound signal and routing the sound signal to your auditory cortex and that's what's allowing you to understand my words. Neuroscientists have done the following fascinating experiments where you cut the wire from the ears to the auditory cortex and you re-wire, in this case an animal's brain, so that the signal from the eyes to the optic nerve eventually gets routed to the auditory cortex. If you do this it turns out, the auditory cortex will learn to see. And this is in every single sense of the word see as we know it. So, if you do this to the animals, the animals can perform visual discrimination task and as they can look at images and make appropriate decisions based on the images and they're doing it with that piece of brain tissue. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/8.png)

Here's another example. That red piece of brain tissue is your somatosensory cortex. That's how you process your sense of touch. If you do a similar re-wiring process then the somatosensory cortex will learn to see. Because of this and other similar experiments, these are called neuro-rewiring experiments. There's this sense that if the same piece of physical brain tissue can process sight or sound or touch then maybe there is one learning algorithm that can process sight or sound or touch. And instead of needing to implement a thousand different programs or a thousand different algorithms to do, you know, the thousand wonderful things that the brain does, maybe what we need to do is figure out some approximation or to whatever the brain's learning algorithm is and implement that and that the brain learned by itself how to process these different types of data. To a surprisingly large extent, it seems as if we can plug in almost any sensor to almost any part of the brain and so, within the reason, the brain will learn to deal with it.

#### Sensor representations in the brain

Here are a few more examples. On the upper left is an example of learning to see with your tongue. The way it works is--this is actually a system called BrainPort undergoing FDA trials now to help blind people see--but the way it works is, you strap a grayscale camera to your forehead, facing forward, that takes the low resolution grayscale image of what's in front of you and you then run a wire to an array of electrodes that you place on your tongue so that each pixel gets mapped to a location on your tongue where maybe a high voltage corresponds to a dark pixel and a low voltage corresponds to a bright pixel and, even as it does today, with this sort of system you and I will be able to learn to see, you know, in tens of minutes with our tongues.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/9.png)

Here's a second example of human echo location or human sonar. So there are two ways you can do this. You can either snap your fingers, or click your tongue. I can't do that very well. But there are blind people today that are actually being trained in schools to do this and learn to interpret the pattern of sounds bouncing off your environment - that's sonar. So, if after you search on YouTube, there are actually videos of this amazing kid who tragically because of cancer had his eyeballs removed, so this is a kid with no eyeballs. But by snapping his fingers, he can walk around and never hit anything. He can ride a skateboard. He can shoot a basketball into a hoop and this is a kid with no eyeballs.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/10.png)

Third example is the Haptic Belt where if you have a strap around your waist, ring up buzzers and always have the northmost one buzzing. You can give a human a direction sense similar to maybe how birds can, you know, sense where north is.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/11.png)

And, some of the bizarre example, but if you plug a third eye into a frog, the frog will learn to use that eye as well. 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/12.png)

**So, it's pretty amazing to what extent is as if you can plug in almost any sensor to the brain and the brain's learning algorithm will just figure out how to learn from that data and deal with that data. And there's a sense that if we can figure out what the brain's learning algorithm is, and, you know, implement it or implement some approximation to that algorithm on a computer, maybe that would be our best shot at, you know, making real progress towards the AI, the artificial intelligence dream of someday building truly intelligent machines.** Now, of course, I'm not teaching Neural Networks, you know, just because they might give us a window into this far-off AI dream, even though I'm personally, that's one of the things that I personally work on in my research life. But the main reason I'm teaching Neural Networks in this class is because it's actually a very effective state of the art technique for modern day machine learning applications. So, in the next few videos, we'll start diving into the technical details of Neural Networks so that you can apply them to modern-day machine learning applications and get them to work well on problems. But for me, you know, one of the reasons the excite me  is that maybe they give us this window into what we might do if we're also thinking of what algorithms might someday be able to learn in a manner similar to humankind.

## neural-networks

### Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (**dendrites**) as electrical inputs (called "spikes") that are channeled to outputs (**axons**).

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/13.png)

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/14.png)

In our model, our dendrites are like the input features $x_1⋯x_n​$, and the output is the result of our hypothesis function. In this model our $x_0​$ input node is sometimes called the "**bias unit**". It is always equal to 1. In neural networks, we use the same logistic function as in classification,  $\frac{1}{1+e^{−θ^Tx}}​$, yet we sometimes call it a sigmoid (logistic) **activation** function. In this situation, our "theta" parameters are sometimes called "weights". 
![Neuron-model_Logistic-unit](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/15.png)

Visually, a simplistic representation looks like:
$$
\begin{bmatrix}x_0 \newline x_1 \\ x_2 \\ \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \\ \end{bmatrix}\rightarrow h_\theta(x)
$$
Our input nodes (layer 1), also known as the "**input layer**", go into another node (layer 2), which finally outputs the hypothesis function, known as the "**output layer**". We can have intermediate layers of nodes between the input and output layers called the "**hidden layers**". 

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/16.png)



In this example, we label these intermediate or *"hidden" layer* nodes $a^{(2)}_0 ⋯ a^{(2)}_n$ and call them "**activation units**".
$$
{% raw %}\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \\ & \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}{% endraw %}
$$
If we had one hidden layer, it would look like:
$$
\begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \\ \end{bmatrix}\rightarrow h_\theta(x)
$$
The values for each of the "activation" nodes is obtained as follows:
$$
{% raw %}\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\ a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{align*}{% endraw %}
$$
This is saying that we compute our activation nodes by using a $3×4​$ matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $Θ^{(2)}​$ containing the weights for our second layer of nodes. 

Each layer gets its own matrix of weights, $Θ^{(j)}$. The dimensions of these matrices of weights is determined as follows: If network has $s_j$ units in layer j and $s_{j+1}$ units in layer j+1, then $Θ^{(j)}$ will be of dimension $s_{j+1}×(s_j+1)$.If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer j+1, then $Θ^(j)$ will be of dimension $s_{j+1}×(s_j+1)$. ***The +1 comes from the addition in $Θ^{(j)}$ of the "bias nodes," $x_0$ and $Θ^{(j)}_0$. In other words the output nodes will not include the bias nodes while the inputs will.***  The following image summarizes our model representation:

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/17.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $Θ^{(1)}$ is going to be $4×3$ where $s_j=2$ and $s_j+1=4$ so $s_{j+1}×(sj+1)=4×3$ .

### Model Representation II

To re-iterate, the following is an example of a neural network:
$$
{% raw %}\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\ a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{align*}{% endraw %}
$$
In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $z^{(j)}_k$ that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:
$$
{% raw %}\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \\ a_2^{(2)} = g(z_2^{(2)}) \\ a_3^{(2)} = g(z_3^{(2)}) \\ \end{align*}{% endraw %}
$$
In other words, for layer j=2 and node k, the variable z will be:
$$
z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
$$
The vector representation of $x$ and $z_j$ is :
$$
{% raw %}\begin{align*}x = \begin{bmatrix}x_0 \\ x_1 \\\cdots \\ x_n\end{bmatrix},  z^{(j)} = \begin{bmatrix}z_1^{(j)} \\ z_2^{(j)} \\\cdots \\ z_n^{(j)}\end{bmatrix}\end{align*}{% endraw %}
$$
Setting $x=a^{(1)}$, we can rewrite the equation as:
$$
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
$$
We are multiplying our matrix $Θ^{(j−1)}$ with dimensions $s_j×(n+1)$ (where $s_j$ is the number of our activation nodes) by our vector $a^{(j−1)}$ with height $(n+1)$. This gives us our vector $z(j)$ with height $s_j$. Now we can get a vector of our activation nodes for layer $j$ as follows:
$$
a^{(j)} = g(z^{(j)})
$$
Where our function g can be applied element-wise to our vector $z^{(j)}$. 

We can then add a bias unit (equal to 1) to layer j after we have computed $a^{(j)}$. This will be element $a^{(j)}_0$ and will be equal to 1. To compute our final hypothesis, let's first compute another $z$ vector:
$$
z^{(j+1)} = \Theta^{(j)}a^{(j)}
$$
We get this final $z$ vector by multiplying the next theta matrix after $Θ^{(j−1)}$ with the values of all the activation nodes we just got. This last theta matrix $Θ^{(j)}$ will have only **one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. We then get our final result with:
$$
h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})
$$
![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/18.png)

Notice that in this **last step**, between layer j and layer j+1, we are doing **exactly the same thing** as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

#### Neural network learning its own features

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/19.png)

let's say I cover up the left path of this picture for now. If you look at what's left in this picture. This looks a lot like logistic regression where what we're doing is we're using that note, that's just the logistic regression unit and we're using that to make a prediction h of x. And concretely, what the hypotheses is outputting is h of x is going to be equal to g which is my sigmoid activation function times theta 0 times a0 is equal to 1 plus theta 1 plus theta 2 times a2 plus theta 3 times a3 whether values a1, a2, a3 are those given by these three given units. Now, to be actually consistent to my early notation.

Actually, we need to, you know, fill in these superscript 2's here everywhere and I also have these indices 1 there because I have only one output unit, but if you focus on the blue parts of the notation. This is, you know, this looks awfully like the standard logistic regression model, except that I now have a capital theta instead of lower case theta. And what this is doing is just logistic regression. But where the features fed into logistic regression are these values computed by the hidden layer. Just to say that again, what this neural network is doing is just like logistic regression, except that rather than using the original features x1, x2, x3, is using these new features a1, a2, a3. Again, we'll put the superscripts there, you know, to be consistent with the notation. And the cool thing about this, is that the features a1, a2, a3, they themselves are learned as functions of the input. Concretely, the function mapping from layer 1 to layer 2, that is determined by some other set of parameters, theta 1. So it's as if the neural network, instead of being constrained to feed the features x1, x2, x3 to logistic regression. It gets to learn its own features, a1, a2, a3, to feed into the logistic regression and as you can imagine depending on what parameters it chooses for theta 1. 
You can learn some pretty interesting and complex features and therefore you can end up with a better hypotheses than if you were constrained to use the raw features x1, x2 or x3 or if you will constrain to say choose the polynomial terms, you know, x1, x2, x3, and so on. But instead, this algorithm has the flexibility to try to learn whatever features at once, using these a1, a2, a3 in order to feed into this last unit that's essentially a logistic regression here. 
I realized this example is described as a somewhat high level and so I'm not sure if this intuition of the neural network, you know, having more complex features will quite make sense yet, but if it doesn't yet in the next two videos I'm going to go through a specific example of how a neural network can use this hidden there to compute more complex features to feed into this final output layer and how that can learn more complex hypotheses. So, in case what I'm saying here doesn't quite make sense, stick with me for the next two videos and hopefully out there working through those examples this explanation will make a little bit more sense. But just the point O. 

#### Other network architectures

You can have neural networks with other types of diagrams as well, and the way that neural networks are connected, that's called the architecture. So the term architecture refers to how the different neurons are connected to each other. This is an example of a different neural network architecture and once again you may be able to get this intuition of how the second layer, here we have three heading units that are computing some complex function maybe of the input layer, and then the third layer can take the second layer's features and compute even more complex features in layer three so that by the time you get to the output layer, layer four, you can have even more complex features of what you are able to compute in layer three and so get very interesting nonlinear hypotheses. By the way, in a network like this, layer one, this is called an input layer. Layer four is still our output layer, and this network has two hidden layers.

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/20.png)

So anything that's not an input layer or an output layer is called a hidden layer. So, hopefully from this video you've gotten a sense of how the feed forward propagation step in a neural network works where you start from the activations of the input layer and forward propagate that to the first hidden layer, then the second hidden layer, and then finally the output layer. And you also saw how we can vectorize that computation. In the next, I realized that some of the intuitions in this video of how, you know, other certain layers are computing complex features of the early layers. I realized some of that intuition may be still slightly abstract and kind of a high level. And so what I would like to do in the two videos is work through a detailed example of how a neural network can be used to compute nonlinear functions of the input and hope that will give you a good sense of the sorts of complex nonlinear hypotheses we can get out of Neural Networks.

## Applications

### Examples and Intuitions I

A simple example of applying neural networks is by predicting x1 AND x2, which is the logical 'and' operator and is only true if both x1 and x2 are 1. 

The graph of our functions will look like:
$$
{% raw %}\begin{align*}\begin{bmatrix}x_0 \\ x_1 \\ x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}{% endraw %}
$$
Remember that x0 is our bias variable and is always 1. 

Let's set our first theta matrix as:
$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix}
$$
This will cause the output of our hypothesis to only be positive if both x1 and x2 are 1. In other words:
$$
{% raw %}\begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \\ \\ & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \\ & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \\ & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \\ & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*}{% endraw %}
$$
![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/21.png)

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. 

Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either x1 is true or x2 is true, or both:

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/22.png)

### Examples and Intuitions II

The $Θ^{(1)}$ matrices for AND, NOR, and OR are :
$$
{% raw %}\begin{align*}AND:\\ \Theta^{(1)} &=\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \\ NOR:\\ \Theta^{(1)} &= \begin{bmatrix}10 & -20 & -20\end{bmatrix} \\ OR:\\ \Theta^{(1)} &= \begin{bmatrix}-10 & 20 & 20\end{bmatrix} \\ \end{align*}{% endraw %}
$$
We can combine these to get the XNOR logical operator (which gives 1 if $x_1$ and $x_2$ are both 0 or both 1).
$$
{% raw %}\begin{align*}\begin{bmatrix}x_0 \\ x_1 \\ x_2\end{bmatrix} \rightarrow\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \end{bmatrix} \rightarrow\begin{bmatrix}a^{(3)}\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}{% endraw %}
$$
For the transition between the first and second layer, we'll use a $Θ^{(1)}$ matrix that combines the values for AND and NOR:
$$
\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20 \\ 10 & -20 & -20\end{bmatrix}
$$
For transition between the second and third layer, we'll use a $Θ^{(2)}$ matrix that uses the value for OR:
$$
\Theta^{(2)} =\begin{bmatrix}-10 & 20 & 20\end{bmatrix}
$$
Let's write out the values for all our nodes:
$$
{% raw %}\begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \\ & a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \\ & h_\Theta(x) = a^{(3)}\end{align*}{% endraw %}
$$
And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/23.png)

### Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:

![](http://q83p23d9i.bkt.clouddn.com/gitpage/ml-andrew-ng/08/24.png)

We can define our set of resulting classes as y:
$$
y^{(i)}=\begin{bmatrix}1\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\0\\1\end{bmatrix}
$$
Each $y^{(i)}$ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like: 
$$
\begin{bmatrix}x_0\\x_1\\x_2\\ \cdots \\ x_n\end{bmatrix} \rightarrow \begin{bmatrix}a_0^{(2)}\\a_1^{(2)}\\ \cdots \\ a_n^{(2)}\end{bmatrix} \rightarrow \rightarrow \begin{bmatrix}a_0^{(3)}\\a_1^{(3)}\\ \cdots \\ a_n^{(3)}\end{bmatrix} \rightarrow \cdots \rightarrow \begin{bmatrix}h_\Theta(x)_1\\h_\Theta(x)_2\\h_\Theta(x)_3\\h_\Theta(x)_4\end{bmatrix}
$$
Our resulting hypothesis for one set of inputs may look like: 
$$
h_\Theta(x) =\begin{bmatrix}0 \\ 0 \\ 1 \\ 0 \\ \end{bmatrix}
$$
In which case our resulting class is the third one down, or $h_Θ(x)_3$, which represents the motorcycle.

## Word Dict


2. pertain to : relate to  与 ... 相关

3. cortex 皮层；皮质；(尤指)大脑皮层

4. optic 
   ◙ adj. [usually before noun]
   • (technical 术语) connected with the eye or the sense of sight
   • 眼的；视觉的:
    »the optic nerve (= from the eye to the brain) 
     视神经

5. route

   - ◙ verb 
     • [VN , usually +adv. / prep.] to send sb / sth by a particular route
     • 按某路线发送:
      »Satellites route data all over the globe. 
       衞星向全球各地传递信息。

6. wire 

   - ~ sb / sth up (to sth) | ~ sb / sth to sth 

     to connect sb / sth to a piece of equipment, especially a tape recorder or computer system

     • 将…连接到(磁带录音机、计算机等设备):

      »He was wired up to a police tape recorder. 他被连接到了警方的录音机上。 

   - [C, U] a piece of wire that is used to carry an electric current or signal
     • 电线；导线:
      »overhead wires 
     架空电线 
      »fuse wire 
     保险丝 
      »The telephone wires had been cut. 
     电话线被割断了。 

7. tissue 
   ◙ noun 

   1. [U] (also tis∙sues [pl.]) a collection of cells that form the different parts of humans, animals and plants • (人、动植物细胞的)组织:
      »muscle / brain / nerve, etc. tissue 
       肌肉、大脑、神经等组织 
      »scar tissue 
       瘢痕组织 
   2. [C] a piece of soft paper that absorbs liquids, used especially as a handkerchief  • (尤指用作手帕的)纸巾,手巾纸:
      »a box of tissues 
       一盒纸巾 
   3. (also 'tissue paper) [U] very thin paper used for wrapping and packing things that break easily • (用于包装易碎物品的)薄纸,绵纸
        【IDIOMS】
          ◘ a tissue of 'lies 
          •(literary) a story, an excuse, etc. that is full of lies
          • 一派谎言

8. far-off : 遥远的

9. dendrites 

   ◙ noun 
   • (biology 生) a short branch at the end of a nerve cell, which receives signals from other cells
   • 树突(位于神经元末端的细分支,接收其他神经元传来的冲动)

10. axon

    ◙ noun 
       • (biology 生) the long thin part of a nerve cell along which signals are sent to other cells
       • 轴突(神经细胞的突起,将信号发送到其他细胞)

11. propagate 传播，宣传





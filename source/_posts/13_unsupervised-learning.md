---
title: 13_unsupervised-learning note13
date: 2018-01-13
copyright: true
categories: English
tags: [Machine Learning]
mathjax: true
mathjax2: true
---

## Note

This personal note is written after studying the opening course on [the coursera website](https://www.coursera.org), [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

## 01_clustering

### 01_unsupervised-learning-introduction

In this video, I'd like to start to talk about **clustering**. This will be exciting, because this is our first unsupervised learning algorithm, where we learn from unlabeled data instead from labelled data. 

So, what is unsupervised learning? I briefly talked about unsupervised learning at the beginning of the class but it's useful to contrast it with supervised learning. So, here's a typical supervised learning problem where we're given a labeled training set and the goal is to find the decision boundary that separates the positive label examples and the negative label examples. So, the supervised learning problem in this case is given a set of labels to fit a hypothesis to it.



![supervised_learning_vs_unsupervised_learning1](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/1.png)



In contrast, in the unsupervised learning problem we're given data that does not have any labels associated with it. So, we're given data that looks like this.

![supervised_learning_vs_unsupervised_learning2](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/2.png)



Here's a set of points add in no labels, and so, our training set is written just x1, x2, and so on up to xm and we don't get any labels y. And that's why the points plotted up on the figure don't have any labels with them. **So, in unsupervised learning what we do is we give this sort of unlabeled training set to an algorithm and we just ask the algorithm find some structure in the data for us.** 

Given this data set one type of structure we might have an algorithm find is that it looks like this data set has points grouped into two separate clusters and so an algorithm that finds clusters like the ones I've just circled is called ***a clustering algorithm***. And **this would be our first type of unsupervised learning, although there will be other types of unsupervised learning algorithms that we'll talk about later that finds other types of structure or other types of patterns in the data other than clusters.** We'll talk about this after we've talked about clustering. 

So, what is clustering good for? Early in this class I already mentioned a few applications. One is market segmentation where you may have a database of customers and want to group them into different marker segments so you can sell to them separately or serve your different market segments better. Social network analysis. There are actually groups have done this things like looking at a group of people's social networks. So, things like Facebook, Google+, or maybe information about who other people that you email the most frequently and who are the people that they email the most frequently and to find coherence in groups of people. So, this would be another maybe clustering algorithm where you know want to find who are the coherent groups of friends in the social network? Here's something that one of my friends actually worked on which is, use clustering to organize computer clusters or to organize data centers better. Because if you know which computers in the data center in the cluster tend to work together, you can use that to reorganize your resources and how you layout the network and how you design your data center communications. And lastly, something that actually another friend worked on using clustering algorithms to understand galaxy formation and using that to understand astronomical data. 

![Applications_of_clustering](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/3.png)



So, that's clustering which is our first example of an unsupervised learning algorithm. In the next video we'll start to talk about a specific clustering algorithm.

#### summary

Unsupervised learning is contrasted from supervised learning because it uses an **unlabeled** training set rather than a labeled one. 
In other words, we don't have the vector y of expected results, we only have a dataset of features where we can find structure. 
Clustering is good for:

* Market segmentation 
* Social network analysis 
* Organizing computer clusters 
* Astronomical data analysis

### 02_k-means-algorithm

In the clustering problem we are given an unlabeled data set and we would like to have an algorithm automatically group the data into coherent subsets or into coherent clusters for us. **The K Means algorithm is by far the most popular, by far the most widely used clustering algorithm**, and in this video I would like to tell you what the K Means Algorithm is and how it works. The K means clustering algorithm is best illustrated in pictures. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/4.png)



Let's say I want to take an unlabeled data set like the one shown here, and I want to group the data into two clusters. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/5.png)



If I run the K Means clustering algorithm, here is what I'm going to do. The first step is to randomly initialize two points, called the cluster centroids. So, these two crosses here, these are called the **Cluster Centroids** and I have two of them because I want to group my data into two clusters. **K Means is an iterative algorithm and it does two things. First is a cluster assignment step, and second is a move centroid step.** 

So, let me tell you what those things mean. **The first of the two steps** in the loop of K means, is this ***cluster assignment step***. What that means is that, it's going through each of the examples, each of these green dots shown here and depending on whether it's closer to the red cluster centroid or the blue cluster centroid, it is going to assign each of the data points to one of the two cluster centroids. Specifically, what I mean by that, is to go through your data set and color each of the points either red or blue, depending on whether it is closer to the red cluster centroid or the blue cluster centroid, and I've done that in this diagram here. So, that was the cluster assignment step. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/6.png)



The other part of K means, in the loop of K means, is the **move centroid step**, and what we are going to do is, we are going to take the two cluster centroids, that is, the red cross and the blue cross, and we are going to move them to the average of the points colored the same colour. So what we are going to do is look at all the red points and compute the average, really the mean of the location of all the red points, and we are going to move the red cluster centroid there. And the same things for the blue cluster centroid, look at all the blue dots and compute their mean, and then move the blue cluster centroid there. So, let me do that now. We're going to move the cluster centroids as follows and I've now moved them to their new means. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/7.png)

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/8.png)


The red one moved like that and the blue one moved like that and the red one moved like that. And then we go back to another cluster assignment step, so we're again going to look at all of my unlabeled examples and depending on whether it's closer the red or the blue cluster centroid, I'm going to color them either red or blue. I'm going to assign each point to one of the two cluster centroids, so let me do that now. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/9.png)


And so the colors of some of the points just changed. And then I'm going to do another move centroid step. So I'm going to compute the average of all the blue points, compute the average of all the red points and move my cluster centroids like this, and so, let's do that again. Let me do one more cluster assignment step. So colour each point red or blue, based on what it's closer to and then do another move centroid step and we're done. 
![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/10.png)


![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/11.png)


![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/12.png)



And in fact if you keep running additional iterations of K means from here the cluster centroids will not change any further and the colours of the points will not change any further. And so, this is the, at this point, K means has converged and it's done a pretty good job finding the two clusters in this data. 

Let's write out the K means algorithm more formally. **The K means algorithm takes two inputs. One is a parameter K,** which is the number of clusters you want to find in the data. I'll later say how we might go about trying to choose k, but for now let's just say that we've decided we want a certain number of clusters and we're going to tell the algorithm how many clusters we think there are in the data set. **And then K means also takes as input this sort of unlabeled training set of just the Xs and because this is unsupervised learning, we don't have the labels Y anymore.** 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/13.png)


And **for unsupervised learning of the K means I'm going to use the ** *convention* **that $X^{(i)}$ is an $R^N$ dimensional vector. And that's why my training examples are now N dimensional rather N plus one dimensional vectors.** This is what the K means algorithm does. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/14.png)


**The first step is that it randomly initializes k cluster centroids which we will call mu 1, mu 2, up to mu k.** And so in the earlier diagram, the cluster centroids corresponded to the location of the red cross and the location of the blue cross. So there we had two cluster centroids, so maybe the red cross was mu 1 and the blue cross was mu 2, and more generally we would have k cluster centroids rather than just 2. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/15.png)



**Then the inner loop of k means does the following**, we're going to repeatedly do the following. First for each of my training examples, I'm going to set this variable $c^{(i)}$ to be the index 1 through K of the cluster centroid closest to $x^{(i)}$. So this was my **cluster assignment step**, where we took each of my examples and coloured it either red or blue, depending on which cluster centroid it was closest to. So $c^{(i)}$ is going to be a number from 1 to K that tells us, you know, is it closer to the red cross or is it closer to the blue cross, and another way of writing this is I'm going to, to compute $c^{(i)}$, I'm going to take my $i_{th}$ example $x^{(i)}$ and and I'm going to **measure it's distance to each of my cluster centroids**, this is mu and then lower-case k, right, so capital K is the total number centroids and I'm going to use lower case k here to index into the different centroids. But so, **$c^{(i)}$ is going to, I'm going to minimize over my values of k and find the value of K that minimizes this distance between Xi and the cluster centroid, and then, you know, the value of k that minimizes this, that's what gets set in $c^{(i)}$.** So, here's another way of writing out what Ci is. If I write the norm between Xi minus Mu-k, then this is the distance between my ith training example Xi and the cluster centroid Mu subscript K, this is--this here, that's a lowercase K. So uppercase K is going to be used to denote the total number of cluster centroids, and this lowercase K's a number between one and capital K. I'm just using lower case K to index into my different cluster centroids. Next is lower case k. So that's the distance between the example and the cluster centroid and so what I'm going to do is find the value of K, of lower case k that minimizes this, and so the value of k that minimizes you know, that's what I'm going to set as Ci, and by convention here I've written the distance between Xi and the cluster centroid, by convention people actually tend to write this as the squared distance. So we think of Ci as picking the cluster centroid with the smallest squared distance to my training example Xi. But of course minimizing squared distance, and minimizing distance that should give you the same value of Ci, but we usually put in the square there, just as the convention that people use for K means. So that was the cluster assignment step. 
![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/16.png)


The other in the loop of K means does the **move centroid step**. And what that does is for each of my cluster centroids, so for lower case k equals 1 through K, **it sets Mu-k equals to the average of the points assigned to the cluster**. So as a concrete example, let's say that one of my cluster centroids, let's say cluster centroid two, has training examples, you know, 1, 5, 6, and 10 assigned to it. And what this means is, really this means that C1 equals to C5 equals to C6 equals to and similarly well c10 equals, too, right? If we got that from the cluster assignment step, then that means examples 1,5,6 and 10 were assigned to the cluster centroid two. Then in this move centroid step, what I'm going to do is just compute the average of these four things. So X1 plus X5 plus X6 plus X10. And now I'm going to average them so here I have four points assigned to this cluster centroid, just take one quarter of that. And now Mu2 is going to be an n-dimensional vector. Because each of these example x1, x5, x6, x10 each of them were an n-dimensional vector, and I'm going to add up these things and, you know, divide by four because I have four points assigned to this cluster centroid, I end up with my move centroid step, for my cluster centroid mu-2. This has the effect of moving mu-2 to the average of the four points listed here. One thing that I've asked is, well here we said, let's let mu-k be the average of the points assigned to the cluster. 

**But what if there is a cluster centroid no points with zero points assigned to it. In that case the more common thing to do is to just eliminate that cluster centroid. And if you do that, you end up with K minus one clusters instead of k clusters. Sometimes if you really need k clusters, then the other thing you can do if you have a cluster centroid with no points assigned to it is you can just randomly reinitialize that cluster centroid, but it's more common to just eliminate a cluster if somewhere during K means it with no points assigned to that cluster centroid, and that can happen, altthough in practice it happens not that often.** So that's the K means Algorithm. 

Before wrapping up this video I just want to tell you about one other common application of K Means and that's to the problems with non well separated clusters. Here's what I mean. So far we've been picturing K Means and applying it to data sets like that shown here where we have three pretty well separated clusters, and we'd like an algorithm to find maybe the 3 clusters for us. **But it turns out that very often K Means is also applied to data sets that look like this where there may not be several very well separated clusters.**

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/17.png)


Here is an example application, to t-shirt sizing. Let's say you are a t-shirt manufacturer you've done is you've gone to the population that you want to sell t-shirts to, and you've collected a number of examples of the height and weight of these people in your population and so, well I guess height and weight tend to be positively highlighted so maybe you end up with a data set like this, you know, with a sample or set of examples of different peoples heights and weight. Let's say you want to size your t shirts. Let's say I want to design and sell t shirts of three sizes, small, medium and large. So how big should I make my small one? How big should I my medium? And how big should I make my large t-shirts. One way to do that would to be to run my k means clustering logarithm on this data set that I have shown on the right and maybe what K Means will do is group all of these points into one cluster and group all of these points into a second cluster and group all of those points into a third cluster. So, even though the data, you know, before hand it didn't seem like we had 3 well separated clusters, K Means will kind of separate out the data into multiple pluses for you. And what you can do is then look at this first population of people and look at them and, you know, look at the height and weight, and try to design a small t-shirt so that it kind of fits this first population of people well and then design a medium t-shirt and design a large t-shirt. And this is in fact kind of an example of market segmentation where you're using K Means to separate your market into 3 different segments. So you can design a product separately that is a small, medium, and large t-shirts, that tries to suit the needs of each of your 3 separate sub-populations well. So that's the K Means algorithm. And by now you should know how to implement the K Means Algorithm and kind of get it to work for some problems. 

But in the next few videos what I want to do is really get more deeply into the nuts and bolts of K means and to talk a bit about how to actually get this to work really well.

#### summary

The K-Means Algorithm is the most popular and widely used algorithm for automatically grouping data into coherent subsets. 

1. Randomly initialize two points in the dataset called the cluster centroids . 
2. Cluster assignment: assign all examples into one of two groups based on which cluster centroid the example is closest to. 
3. Move centroid: compute the averages for all the points inside each of the two cluster centroid groups, then move the cluster centroid points to those averages. 
4. Re-run (2) and (3) until we have found our clusters. 

Our main variables are: 
* K (number of clusters) 
* Training set ${x^{(1)}, x^{(2)}, \dots,x^{(m)}}$ 
* Where $x^{(i)} \in \mathbb{R}^n$ 
* Note that we will not use the $x_0=1$ convention. 

**The algorithm:**
```c
Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K)
Repeat:
   for i = 1 to m:
      c(i):= index (from 1 to K) of cluster centroid closest to x(i)
   for k = 1 to K:
      mu(k):= average (mean) of points assigned to cluster k
```

The **first for-loop** is the 'Cluster Assignment' step. We make a vector c where $c^{(i)}$ represents the centroid assigned to example $x^{(i)}$ . 
We can write the operation of the Cluster Assignment step more mathematically as follows: 
$c^{(i)} = argmin_k\ ||x^{(i)} - \mu_k||^2$ 
That is, each $c^{(i)}$ contains the index of the centroid that has minimal distance to $x^{(i)}$. 
By convention, we square the right-hand-side, which makes the function we are trying to minimize more sharply increasing. It is mostly just a convention. But a convention that helps reduce the computation load because the Euclidean distance requires a square root but it is canceled. 
Without the square: 
$$||x^{(i)} - \mu_k|| = ||\quad\sqrt{(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...}\quad||$$ 
With the square: 
$$||x^{(i)} - \mu_k||^2 = ||\quad(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...\quad||$$ 
so the square convention serves two purposes, minimize more sharply and less computation. 
The **second for-loop** is the 'Move Centroid' step where we move each centroid to the average of its group. 
More formally, the equation for this loop is as follows: 
$$\mu_k = \dfrac{1}{n}[x^{(k_1)} + x^{(k_2)} + \dots + x^{(k_n)}] \in \mathbb{R}^n$$ 
Where each of $x^{(k_1)}, x^{(k_2)}, \dots, x^{(k_n)}$ are the training examples assigned to group $mμ_k$. 
If you have a cluster centroid with 0 points assigned to it, you can randomly **re-initialize** that centroid to a new point. You can also simply **eliminate** that cluster group. 
After a number of iterations the algorithm will **converge** , where new iterations do not affect the clusters. 
Note on non-separated clusters: some datasets have no real inner separation or natural structure. K-means can still evenly segment your data into K subsets, so can still be useful in this case. 

### 03_optimization-objective

Most of the supervised learning algorithms we've seen, things like linear regression, logistic regression, and so on, all of those algorithms have an optimization objective or some cost function that the algorithm was trying to minimize. **It turns out that k-means also has an optimization objective or a cost function that it's trying to minimize**. 

And in this video I'd like to tell you what that optimization objective is. And the reason I want to do so is because this will be useful to us for two purposes. **First, knowing what is the optimization objective of k-means will help us to debug the learning algorithm and just make sure that k-means is running correctly. And second, and perhaps more importantly, in a later video we'll talk about how we can use this to help k-means find better costs for this and avoid the local optima.** 

But we do that in a later video that follows this one. **Just as a quick reminder while k-means is running we're going to be keeping track of two sets of variables. First is the $c^{(i)}$'s and that keeps track of the index or the number of the cluster, to which an example $x^{(i)}$ is currently assigned. And then the other set of variables we use is $/mu_k$, which is the location of cluster centroid k.**

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/18.png)

Again, for k-means we use capital K to denote the total number of clusters. And here lower case k is going to be an index into the cluster centroids and so, lower case k is going to be a number between one and capital K. Now here's one more bit of notation, which is gonna use mu subscript ci ($/mu_{c^{(i)}}$) to denote the cluster centroid of the cluster to which example $x^{(i)}$ has been assigned, right? And to explain that notation a little bit more, lets say that xi has been assigned to cluster number five. What that means is that ci, that is the index of xi, that that is equal to five. Right? Because having ci equals five, if that's what it means for the example xi to be assigned to cluster number five. And so mu subscript ci is going to be equal to mu subscript 5. Because ci is equal to five. And so this mu subscript ci is the cluster centroid of cluster number five, which is the cluster to which my example xi has been assigned. 

![K-means_optimization_objective](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/19.png)



Out with this notation, we're now ready to write out what is the optimization objective of the k-means clustering algorithm and here it is. The cost function that k-means is minimizing is a function J of all of these parameters, $c^{(1)}$ through $c^{(m)}$ and $/mu_1$ through $/gmu_K$. That k-means is varying as the algorithm runs. And the optimization objective is shown to the right, is the average of 1 over m of sum from i equals 1 through m of this term here. That I've just drawn the red box around, right? The square distance between each example xi and the location of the cluster centroid to which xi has been assigned. So let's draw this and just let me explain this. Right, so here's the location of training example xi and here's the location of the cluster centroid to which example xi has been assigned. So to explain this in pictures, if here's x1, x2, and if a point here is my example xi, so if that is equal to my example xi, and if xi has been assigned to some cluster centroid, I'm gonna denote my cluster centroid with a cross, so if that's the location of mu 5, let's say. If x i has been assigned cluster centroid five as in my example up there, then this square distance, that's the square of the distance between the point xi and this cluster centroid to which xi has been assigned. And what k-means can be shown to be doing is that it is trying to define parameters ci and mu i. Trying to find c and mu to try to minimize this cost function J. This cost function is sometimes also called the **distortion cost function**, or the distortion of the k-means algorithm. 

![K-means-algorithm_in_optimization_view](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/20.png)



And just to provide a little bit more detail, here's the k-means algorithm. Here's exactly the algorithm as we have written it out on the earlier slide.  **And what this first step of this algorithm is, this was the cluster assignment step where we assigned each point to the closest centroid. And it's possible to show mathematically that what the cluster assignment step is doing is exactly Minimizing J, with respect to the variables c1, c2 and so on, up to cm, while holding the cluster centroids mu 1 up to mu K, fixed.** So what the cluster assignment step does is it doesn't change the cluster centroids, but what it's doing is this is exactly picking the values of c1, c2, up to cm. That minimizes the cost function, or the distortion function J. And it's possible to prove that mathematically, but I won't do so here. But it has a pretty intuitive meaning of just well, let's assign each point to a cluster centroid that is closest to it, because that's what minimizes the square of distance between the points in the cluster centroid. **And then the second step of k-means, this second step over here. The second step was the move centroid step. And once again I won't prove it, but it can be shown mathematically that what the move centroid step does is it chooses the values of mu that minimizes J, so it minimizes the cost function J with respect to, wrt is my abbreviation for, with respect to, when it minimizes J with respect to the locations of the cluster centroids mu 1 through mu K.** So if is really is doing is this taking the two sets of variables and partitioning them into two halves right here. First the c sets of variables and then you have the mu sets of variables. ***And what it does is it first minimizes J with respect to the variable c and then it minimizes J with respect to the variables mu and then it keeps on. And, so all that's all that k-means does.*** And now that we understand k-means as trying to minimize this cost function J, we can also use this to try to debug other any algorithm and just kind of make sure that our implementation of k-means is running correctly. 

So, we now understand the k-means algorithm as trying to optimize this cost function J, which is also called the distortion function. We can use that to debug k means and help make sure that k-means is converging and is running properly. And in the next video we'll also see how we can use this to help k-means find better clusters and to help k-means to avoid local optima.

#### summary

Recall some of the parameters we used in our algorithm: 
$c^{(i)}$ = index of cluster (1,2,...,K) to which example $x^{(i)}$ is currently assigned 
$\mu_k $= cluster centroid k (μk∈ℝn) 
$\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned 
Using these variables we can define our **cost function** : 
$$J(c^{(i)},\dots,c^{(m)},\mu_1,\dots,\mu_K) = \dfrac{1}{m}\sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$ 
Our **optimization objective** is to minimize all our parameters using the above cost function: 
$$min_{c,\mu}\ J(c,\mu)$$ 
That is, we are finding all the values in sets c, representing all our clusters, and μ, representing all our centroids, that will minimize the average of the distances of every training example to its corresponding cluster centroid. 
The above cost function is often called the **distortion** of the training examples. 
In the **cluster assignment step** , our goal is to: 
Minimize J(…) with $c^{(1)},\dots,c^{(m)}$ (holding $\mu_1,\dots,\mu_K$ fixed) 
In the **move centroid step**, our goal is to: 
Minimize J(…) with $\mu_1,\dots,\mu_K$ 
With k-means, **it is not possible for the cost function to sometimes increase** . It should always descend. 

### 04_random-initialization

In this video, I'd like to talk about **how to initialize K-means and more importantly, this will lead into a discussion of how to make K-means avoid local optima as well.** Here's the K-means clustering algorithm that we talked about earlier. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/21.png)



**One step** that we never really talked much about was this step of how you randomly initialize the cluster centroids. 

![random_initialization](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/22.png)



There are few different ways that one can imagine using to randomly initialize the cluster centroids. But, it turns out that there is one method that is much more recommended than most of the other options one might think about. So, let me tell you about that option since it's what often seems to work best. Here's how I usually initialize my cluster centroids. **When running K-means, you should have the number of cluster centroids, K, set to be less than the number of training examples M.** It would be really weird to run K-means with a number of cluster centroids that's, you know, equal or greater than the number of examples you have, right? **So the way I usually initialize K-means is, I would randomly pick k training examples. So, and, what I do is then set $\mu_1$ of $\mu_K$ equal to these k examples**.
 
![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/23.png)



 Let me show you a concrete example. Lets say that k is equal to 2 and so on this example on the right let's say I want to find two clusters. So, what I'm going to do in order to initialize my cluster centroids is, I'm going to randomly pick a couple examples. And let's say, I pick this one and I pick that one. And the way I'm going to initialize my cluster centroids is, I'm just going to initialize my cluster centroids to be right on top of those examples. So that's my first cluster centroid and that's my second cluster centroid, and that's one random initialization of K-means. The one I drew looks like a particularly good one. And sometimes I might get less lucky and maybe I'll end up picking that as my first random initial example, and that as my second one. And here I'm picking two examples because k equals 2. Some we have randomly picked two training examples and if I chose those two then I'll end up with, may be this as my first cluster centroid and that as my second initial location of the cluster centroid. So, that's how you can randomly initialize the cluster centroids. 

![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/24.png)


  
 And so at initialization, your first cluster centroid Mu1 will be equal to x(i) for some randomly value of i and Mu2 will be equal to x(j) for some different randomly chosen value of j and so on, if you have more clusters and more cluster centroid. And sort of the side common. I should say that in the earlier video where I first illustrated K-means with the animation. In that set of slides. Only for the purpose of illustration. I actually used a different method of initialization for my cluster centroids.But the method described on this slide, this is really the recommended way. And the way that you should probably use, when you implement K-means. So, as they suggested perhaps by these two illustrations on the right. You might really guess that K-means can end up converging to different solutions depending on exactly how the clusters were initialized, and so, depending on the random initialization. 
 
![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/25.png)


 
**K-means can end up at different solutions. And, in particular, K-means can actually end up at local optima.** If you're given the data sale like this. Well, it looks like, you know, there are three clusters, and so, if you run K-means and if it ends up at a good local optima this might be really the global optima, you might end up with that cluster ring. But if you had a particularly unlucky, random initialization, K-means can also get stuck at different local optima. So, in this example on the left it looks like this blue cluster has captured a lot of points of the left and then the they were on the green clusters each is captioned on the relatively small number of points. And so, this corresponds to a bad local optima because it has basically taken these two clusters and used them into 1 and furthermore, has split the second cluster into two separate sub-clusters like so, and it has also taken the second cluster and split it into two separate sub-clusters like so, and so, both of these examples on the lower right correspond to different local optima of K-means and in fact, in this example here, the cluster, the red cluster has captured only a single optima example. And the term local optima, by the way, refers to local optima of this distortion function J, and what these solutions on the lower left, what these local optima correspond to is really solutions where K-means has gotten stuck to the local optima and it's not doing a very good job minimizing this distortion function J. So, if you're worried about K-means getting stuck in local optima, if you want to increase the odds of K-means finding the best possible clustering, like that shown on top here, what we can do, is try multiple, random initializations. **So, instead of just initializing K-means once and hopping that that works, what we can do is, initialize K-means lots of times and run K-means lots of times, and use that to try to make sure we get as good a solution, as good a local or global optima as possible.**
 
![](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/26.png)



 Concretely, here's how you could go about doing that. Let's say, I decide to run K-meanss a hundred times so I'll execute this loop a hundred times and it's fairly typical a number of times when came to will be something from 50 up to may be 1000. So, let's say you decide to say K-means one hundred times. So what that means is that we would randomnly initialize K-means. And for each of these one hundred random intializations we would run K-means and that would give us a set of clusteringings, and a set of cluster centroids, and then we would then compute the distortion J, that is compute this cause function on the set of cluster assignments and cluster centroids that we got. Finally, having done this whole procedure a hundred times. **You will have a hundred different ways of clustering the data and then finally what you do is all of these hundred ways you have found of clustering the data, just pick one, that gives us the lowest cost. That gives us the lowest distortion. And it turns out that if you are running K-means with a fairly small number of clusters , so you know if the number of clusters is anywhere from two up to maybe 10 - then doing multiple random initializations can often, can sometimes make sure that you find a better local optima. Make sure you find the better clustering data. But if K is very large, so, if K is much greater than 10, certainly if K were, you know, if you were trying to find hundreds of clusters, then, having multiple random initializations is less likely to make a huge difference and there is a much higher chance that your first random initialization will give you a pretty decent solution already and doing, doing multiple random initializations will probably give you a slightly better solution but, but maybe not that much.** But it's really in the regime of where you have a relatively small number of clusters, especially if you have, maybe 2 or 3 or 4 clusters that random initialization could make a huge difference in terms of making sure you do a good job minimizing the distortion function and giving you a good clustering. So, that's K-means with random initialization. If you're trying to learn a clustering with a relatively small number of clusters, 2, 3, 4, 5, maybe, 6, 7, using multiple random initializations can sometimes, help you find much better clustering of the data. But, even if you are learning a large number of clusters, the initialization, the random initialization method that I describe here. That should give K-means a reasonable starting point to start from for finding a good set of clusters.

#### summary

There's one particular recommended method for randomly initializing your cluster centroids. 

1. Have K<m. That is, make sure the number of your clusters is less than the number of your training examples. 
2. Randomly pick K training examples. (Not mentioned in the lecture, but also be sure the selected examples are unique). 
3. Set $\mu_1,\dots,\mu_K$ equal to these K examples. 

**K-means can get stuck in local optima**. To decrease the chance of this happening, you can run the algorithm on many different random initializations. In cases where K<10 it is strongly recommended to run a loop of random initializations. 

```c
for i = 1 to 100:
   randomly initialize k-means
   run k-means to get 'c' and 'm'
   compute the cost function (distortion) J(c,m)
pick the clustering that gave us the lowest cost
```
### 05_choosing-the-number-of-clusters

In this video I'd like to talk about **one last detail of K-means clustering which is how to choose the number of clusters, or how to choose the value of the parameter capsule K.** 

To be honest, there actually isn't a great way of answering this or doing this automatically and by far the most common way of choosing the number of clusters, is still choosing it manually by looking at visualizations or by looking at the output of the clustering algorithm or something else. But I do get asked this question quite a lot of how do you choose the number of clusters, and so I just want to tell you know what are peoples' current thinking on it although, the most common thing is actually to choose the number of clusters by hand. A large part of why it might not always be easy to choose the number of clusters is that it is often generally ambiguous how many clusters there are in the data. Looking at this data set some of you may see four clusters and that would suggest using K equals 4. Or some of you may see two clusters and that will suggest K equals 2 and now this may see three clusters. **And so, looking at the data set like this, the true number of clusters, it actually seems genuinely ambiguous to me, and I don't think there is one right answer. And this is part of our supervised learning. We are aren't given labels, and so there isn't always a clear cut answer.**

![what_is_the_right_of_K](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/27.png)



 And this is one of the things that makes it more difficult to say, have an automatic algorithm for choosing how many clusters to have. When people talk about ways of choosing the number of clusters, one method that people sometimes talk about is something called the **Elbow Method**. Let me just tell you a little bit about that, and then mention some of its advantages but also shortcomings.
 
![elbow_method](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/28.png)



 So the Elbow Method, what we're going to do is vary K, which is the total number of clusters. So, we're going to run K-means with one cluster, that means really, everything gets grouped into a single cluster and compute the cost function or compute the distortion J and plot that here. And then we're going to run K means with two clusters, maybe with multiple random initial agents, maybe not. But then, you know, with two clusters we should get, hopefully, a smaller distortion, and so plot that there. And then run K-means with three clusters, hopefully, you get even smaller distortion and plot that there. I'm gonna run K-means with four, five and so on. And so we end up with a curve showing how the distortion, you know, goes down as we increase the number of clusters. And so we get a curve that maybe looks like this. And if you look at this curve, what the Elbow Method does it says "Well, let's look at this plot. Looks like **there's a clear elbow there**". **Right, this is, would be by analogy to the human arm where, you know, if you imagine that you reach out your arm, then, this is your shoulder joint, this is your elbow joint and I guess, your hand is at the end over here. And so this is the Elbow Method.**

![cost_j_with_elbow](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/29.png)



Then you find this sort of pattern where the distortion goes down rapidly from 1 to 2, and 2 to 3, and then you reach an elbow at 3, and then the distortion goes down very slowly after that. And then it looks like, you know what, maybe using three clusters is the right number of clusters, because that's the elbow of this curve, right? That it goes down, distortion goes down rapidly until K equals 3, really goes down very slowly after that. So let's pick K equals 3. If you apply the Elbow Method, and if you get a plot that actually looks like this, then, that's pretty good, and this would be a reasonable way of choosing the number of clusters. 

It turns out the Elbow Method isn't used that often, and one reason is that, if you actually use this on a clustering problem, it turns out that fairly often, you know, you end up with a curve that looks much more ambiguous, maybe something like this.

![cost_j_without_elbow](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/30.png)



 And if you look at this, I don't know, maybe there's no clear elbow, but it looks like distortion continuously goes down, maybe 3 is a good number, maybe 4 is a good number, maybe 5 is also not bad. And so, if you actually do this in a practice, you know, if your plot looks like the one on the left and that's great. It gives you a clear answer, but just as often, you end up with a plot that looks like the one on the right and is not clear where the ready location of the elbow is. It  makes it harder to choose a number of clusters using this method. So maybe the quick summary of the Elbow Method is that is worth the shot but I wouldn't necessarily, you know, have a very high expectation of it working for any particular problem.
 
 Finally, here's one other way of how, thinking about how you choose the value of K, very often people are running K-means in order you get clusters for some later purpose, or for some sort of downstream purpose. Maybe you want to use K-means in order to do market segmentation, like in the T-shirt sizing example that we talked about. Maybe you want K-means to organize a computer cluster better, or maybe a learning cluster for some different purpose, and so, if that later, downstream purpose, such as market segmentation. If that gives you an evaluation metric, then often, a better way to determine the number of clusters, is to see how well different numbers of clusters serve that later downstream purpose. Let me step through a specific example. 

![choosing_the_value_of_K_for_downstream_purpose](http://q3rrj5fj6.bkt.clouddn.com/gitpage/ml-andrew-ng/13/31.png)



 Let me go through the T-shirt size example again, and I'm trying to decide, do I want three T-shirt sizes? So, I choose K equals 3, then I might have small, medium and large T-shirts. Or maybe, I want to choose K equals 5, and then I might have, you know, extra small, small, medium, large and extra large T-shirt sizes. So, you can have like 3 T-shirt sizes or four or five T-shirt sizes. We could also have four T-shirt sizes, but I'm just showing three and five here, just to simplify this slide for now. So, if I run K-means with K equals 3, maybe I end up with, that's my small and that's my medium and that's my large. Whereas, if I run K-means with 5 clusters, maybe I end up with, those are my extra small T-shirts, these are my small, these are my medium, these are my large and these are my extra large. And the nice thing about this example is that, this then maybe gives us another way to choose whether we want 3 or 4 or 5 clusters, and in particular, what you can do is, you know, think about this from the perspective of the T-shirt business and ask: "Well if I have five segments, then how well will my T-shirts fit my customers and so, how many T-shirts can I sell? How happy will my customers be?" What really makes sense, from the perspective of the T-shirt business, in terms of whether, I want to have Goer T-shirt sizes so that my T-shirts fit my customers better. Or do I want to have fewer T-shirt sizes so that I make fewer sizes of T-shirts. And I can sell them to the customers more cheaply. And so, the t-shirt selling business, that might give you a way to decide, between three clusters versus five clusters. So, that gives you an example of how a later downstream purpose like the problem of deciding what T-shirts to manufacture, how that can give you an evaluation metric for choosing the number of clusters. 
 
 For those of you that are doing the program exercises, if you look at this week's program exercise associative K-means, that's an example there of using K-means for image compression. And so if you were trying to choose how many clusters to use for that problem, you could also, again use the evaluation metric of image compression to choose the number of clusters, K? So, how good do you want the image to look versus, how much do you want to compress the file size of the image, and, you know, if you do the programming exercise, what I've just said will make more sense at that time. 
 
 **So, just summarize, for the most part, the number of customers K is still chosen by hand by human input or human insight. One way to try to do so is to use the Elbow Method, but I wouldn't always expect that to work well, but I think the better way to think about how to choose the number of clusters is to ask, for what purpose are you running K-means? And then to think, what is the number of clusters K that serves that, you know, whatever later purpose that you actually run the K-means for.**

#### summary

Choosing K can be quite arbitrary and ambiguous. 
The **elbow method** : plot the cost J and the number of clusters K. The cost function should reduce as we increase the number of clusters, and then flatten out. Choose K at the point where the cost function starts to flatten out. 
However, fairly often, the curve is very **gradual** , so there's no clear elbow. 
**Note**: J will always decrease as K is increased. The one exception is if k-means gets stuck at a bad local optimum. 
Another way to choose K is to observe how well k-means performs on **a downstream purpose** . In other words, you choose K that proves to be most useful for some goal you're trying to achieve from using these clusters. 

## Bonus: Discussion of the drawbacks of K-Means 
This links to a discussion that shows various situations in which K-means gives totally correct but unexpected results: http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means

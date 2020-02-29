---
title: 01_introduction note1
date: 2018-01-01
copyright: true
categories: English
tags: [Machine Learning]
mathjax: false
mathjax2: false
---

**Need to know**

This personal note is written after studying the coursera opening course, [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. So, the copyright belongs to [Andrew NG](https://en.wikipedia.org/wiki/Andrew_Ng).

## What is Machine Learning?

### Two definitions of Machine Learning are offered

1. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition. 
2. Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." 

**Example: playing checkers.**
E = the experience of playing many games of checkers 
T = the task of playing checkers. 
P = the probability that the program will win the next game. 

**In general, any machine learning problem can be assigned to one of two broad classifications:** 
1. Supervised learning 
2. Unsupervised learning

## Supervised Learning 

**In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.** Supervised learning problems are categorized into "regression" and "classification" problems.
- **In a regression problem**, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. 
- **In a classification problem**, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

### housing price prediction:

**Given data about the size of houses on the real estate market, try to predict their price.** Price as a function of size is a continuous output, so this is a regression problem. 

![housing_price_prediction](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-201851322163.png)

Let's say you want to predict housing prices. A while back, a student collected data sets from the Institute of Portland Oregon. And let's say you plot a data set and it looks like this. Here on the horizontal axis, the size of different houses in square feet, and on the vertical axis, the price of different houses in thousands of dollars. So. Given this data, let's say you have a friend who owns a house that is, say 750 square feet and hoping to sell the house and they want to know how much they can get for the house. So how can the learning algorithm help you? One thing a learning algorithm might be able to do is put a straight line through the data or to fit a straight line to the data and, based on that, it looks like maybe the house can be sold for maybe about 150,000 dollars . But maybe this isn't the only learning algorithm you can use. There might be a better one. For example, instead of sending a straight line to the data, we might decide that it's better to fit a quadratic function or a second-order polynomial to this data. And if you do that, and make a prediction here, then it looks like, well, maybe we can sell the house for closer to $200,000. One of the things we'll talk about later is how to choose and how to decide do you want to fit a straight line to the data or do you want to fit the quadratic function to the data and there's no fair picking whichever one gives your friend the better house to sell. But each of these would be a fine example of a learning algorithm. So this is an example of a supervised learning algorithm.
***We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.*** 

### breast cancer

(a) **Regression** - Given a picture of a person, we have to predict their age on the basis of the given picture 
(b) **Classification** - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 

![breast_cancer](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513223823.png)

If someone discovers a breast tumor, a lump in their breast, a malignant tumor is a tumor that is harmful and dangerous and a benign tumor is a tumor that is harmless. So obviously people care a lot about this. **Let's see you want to look at medical records and try to predict of a breast cancer as malignant or benign.**  
Let's see a collected data set and suppose in your data set you have on your horizontal axis the size of the tumor and on the vertical axis I'm going to plot one or zero, yes or no, whether or not these are examples of tumors we've seen before are malignant which is one or zero if not malignant or benign. 
So let's say our data set looks like this where we saw a tumor of this size that turned out to be benign. One of this size, one of this size. And so on. And sadly we also saw a few malignant tumors, one of that size, one of that size, one of that size... So on. So this example... I have five examples of benign tumors shown down here, and five examples of malignant tumors shown with a vertical axis value of one. 
And let's say we have a friend who tragically has a breast tumor, and let's say her breast tumor size is maybe somewhere around this value. **The machine learning question is, can you estimate what is the probability, what is the chance that a tumor is malignant versus benign? **To introduce a bit more terminology this is an example of a classification problem. The term classification refers to the fact that here we're trying to predict a discrete value output: zero or one, malignant or benign. 
**And it turns out that in classification problems sometimes you can have more than two values for the two possible values for the output.** As a concrete example maybe there are three types of breast cancers and so you may try to predict the discrete value of zero, one, two, or three with zero being benign. Benign tumor, so no cancer. And one may mean, type one cancer, like, you have three types of cancer, whatever type one means. And two may mean a second type of cancer, a three may mean a third type of cancer. But this would also be a classification problem, because this other discrete value set of output corresponding to, you know, no cancer, or cancer type one, or cancer type two, or cancer type three. In classification problems there is another way to plot this data. Let me show you what I mean. 
**Let me use a slightly different set of symbols to plot this data.** So if tumor size is going to be the attribute that I'm going to use to predict malignancy or benignness, I can also draw my data like this. I'm going to use different symbols to denote my benign and malignant, or my negative and positive examples. **So instead of drawing crosses, I'm now going to draw O's for the benign tumors. Like so. And I'm going to keep using X's to denote my malignant tumors. Okay? I hope this is beginning to make sense.** ***（My Note:it isn't a sequential problem, but for the time being, we can ignore it）*** All I did was I took, you know, these, my data set on top and I just mapped it down. To this real line like so. And started to use different symbols, circles and crosses, to denote malignant versus benign examples. 
 **In other machine learning problems when we have more than one feature, more than one attribute.** Here's an example. Let's say that instead of just knowing the tumor size, **we know both the age of the patients and the tumor size.** In that case maybe your data set will look like this where I may have a set of patients with those ages and that tumor size and they look like this.
![Breast_cancer_with_multiple_properties](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513224217.png)

And a different set of patients, they look a little different, whose tumors turn out to be malignant, as denoted by the crosses. So, let's say you have a friend who tragically has a tumor. And maybe, their tumor size and age falls around there. So given a data set like this, what the learning algorithm might do is throw the straight line through the data to try to separate out the malignant tumors from the benign ones and, so the learning algorithm may decide to throw the straight line like that to separate out the two classes of tumors. And. You know, with this, hopefully you can decide that your friend's tumor is more likely to if it's over there, that hopefully your learning algorithm will say that your friend's tumor falls on this benign side and is therefore more likely to be benign than malignant. In this example we had two features, namely, the age of the patient and the size of the tumor. 
**In other machine learning problems we will often have more features**, and my friends that work on this problem, they actually use other features like these, which is clump thickness, the clump thickness of the breast tumor. Uniformity of cell size of the tumor. Uniformity of cell shape of the tumor, and so on, and other features as well. And it turns out one of the interes-, most interesting learning algorithms that we'll see in this class is a learning algorithm that can deal with, not just two or three or five features, but an infinite number of features. On this slide, I've listed a total of five different features. Right, two on the axes and three more up here. But it turns out that for some learning problems, what you really want is not to use, like, three or five features. But instead, you want to use an infinite number of features, an infinite number of attributes, so that **your learning algorithm has lots of attributes or features or cues with which to make those predictions. So how do you deal with an infinite number of features.** How do you even store an infinite number of things on the computer when your computer is gonna run out of memory. **It turns out that when we talk about an algorithm called the Support Vector Machine, there will be a neat mathematical trick that will allow a computer to deal with an infinite number of features.**
### summary
In supervised learning, in every example in our data set, we are told what is the "correct answer" that we would have quite liked the algorithms have predicted on that example. Such as the price of the house, or whether a tumor is malignant or benign. We also talked about the regression problem. And by regression, that means that our goal is to predict a continuous valued output. And we talked about the classification problem, where the goal is to predict a discrete value output.

## Unsupervised Learning

### breast cancer

Back then, recall data sets that look like this, where each example was labeled either as a positive or negative example, whether it was a benign or a malignant tumor. **So for each example in Supervised Learning, we were told explicitly what is the so-called right answer, whether it's benign or malignant.**

![supervised_breast_cancer](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513224410.png)

  **In Unsupervised Learning, we're given data that looks different than data that looks like this that doesn't have any labels or that all has the same label or really no labels. So we're given the data set and we're not told what to do with it and we're not told what each data point is. Instead we're just told, here is a data set. Can you find some structure in the data?** Given this data set, an Unsupervised Learning algorithm might decide that the data lives in two different clusters. And so there's one cluster and there's a different cluster. And yes, Supervised Learning algorithm may break these data into these two separate clusters. So this is called ***a clustering algorithm***. And this turns out to be used in many places.

![unsupervised_learning_of_breasT_cancer](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513224632.png)

### google news

One example where clustering is used is in Google News and if you have not seen this before, you can actually go to this URL news.google.com to take a look. What Google News does is everyday it goes and looks at tens of thousands or hundreds of thousands of new stories on the web and it groups them into cohesive news stories. 

![google_news_cluster1](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513224821.png)

![google_news_cluster2](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513224920.png)

For example, let's look here. The URLs here link to different news stories about the BP Oil Well story. So, let's click on one of these URL's and we'll click on one of these URL's. What I'll get to is a web page like this. Here's a Wall Street Journal article about, you know, the BP Oil Well Spill stories of "BP Kills Macondo", which is a name of the spill and if you click on a different URL from that group then you might get the different story. Here's the CNN story about a game, the BP Oil Spill, and if you click on yet a third link, then you might get a different story. Here's the UK Guardian story about the BP Oil Spill. **So what Google News has done is look for tens of thousands of news stories and automatically cluster them together. So, the news stories that are all about the same topic get displayed together.** It turns out that clustering algorithms and Unsupervised Learning algorithms are used in many other problems as well.

### DNA

Here's one on understanding genomics. Here's an example of DNA microarray data. 
![genes_cluster](http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/what_is_machine_learning_of_DNA.png)
**The idea is put a group of different individuals and for each of them, you measure how much they do or do not have a certain gene. Technically you measure how much certain genes are expressed. So these colors, red, green, gray and so on, they show the degree to which different individuals do or do not have a specific gene. And what you can do is then run a clustering algorithm to group individuals into different categories or into different types of people.** 
So this is Unsupervised Learning because we're not telling the algorithm in advance that these are type 1 people, those are type 2 persons, those are type 3 persons and so on and instead what were saying is yeah here's a bunch of data. I don't know what's in this data. I don't know who's and what type. I don't even know what the different types of people are, but can you automatically find structure in the data from the you automatically cluster the individuals into these types that I don't know in advance? Because we're not giving the algorithm the right answer for the examples in my data set, this is Unsupervised Learning. 
Unsupervised Learning or clustering is used for a bunch of other applications. 

### large computer clusters

It's used to **organize large computer clusters**. I had some friends looking at large data centers, that is large computer clusters and trying to figure out which machines tend to work together and if you can put those machines together, you can make your data center work more efficiently. 

![organize large computer clusters](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513225033.png)

### social network analysis

This second application is on **social network analysis**. So given knowledge about which friends you email the most or given your Facebook friends or your Google+ circles, can we automatically identify which are cohesive groups of friends, also which are groups of people that all know each other? 

![social network analysis](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513225712.png)

### Market segmentation

Many companies have huge databases of customer information. So, can you look at this customer data set and automatically discover market segments and automatically group your customers into different market segments so that you can automatically and more efficiently sell or market your different market segments together? Again, this is Unsupervised Learning because we have all this customer data, but we don't know in advance what are the market segments and for the customers in our data set, you know, we don't know in advance who is in market segment one, who is in market segment two, and so on. But we have to let the algorithm discover all this just from the data. 
![Market segmentation](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-2018513225910.png)

### astronomical data analysis

Finally, it turns out that Unsupervised Learning is also used for surprisingly astronomical data analysis and these clustering algorithms gives surprisingly interesting useful theories of how galaxies are formed.
![astronomical data analysis](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-201851323019.png)
**All of these are examples of clustering, which is just one type of Unsupervised Learning. Let me tell you about another one.**

### cocktail party problem

I'm gonna tell you about the cocktail party problem. So, you've been to cocktail parties before, right? Well, you can imagine there's a party, room full of people, all sitting around, all talking at the same time and there are all these overlapping voices because everyone is talking at the same time, and it is almost hard to hear the person in front of you. So maybe at a cocktail party with two people, two people talking at the same time, and it's a somewhat small cocktail party. **And we're going to put two microphones in the room so there are microphones, and because these microphones are at two different distances from the speakers, each microphone records a different combination of these two speaker voices. Maybe speaker one is a little louder in microphone one and maybe speaker two is a little bit louder on microphone 2 because the 2 microphones are at different positions relative to the 2 speakers, but each microphone would cause an overlapping combination of both speakers' voices.**
![separate_voices_of_cocktail_parties](http://q6gm8fomw.bkt.clouddn.com/e:%5Csnaildove.github.io%5C01-what-is-machine-learning.md-201851323121.png)

 So here's an actual recording of two speakers recorded by a researcher. Let me play for you the first :

<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/1.m4a" controls="true">
Your browser does not support the audio element.
</audio>

what the first microphone sounds like. One (uno), two (dos), three (tres), four (cuatro), five (cinco), six (seis), seven (siete), eight (ocho), nine (nueve), ten (y diez).
 All right, maybe not the most interesting cocktail party, there's two people counting from one to ten in two languages but you know. What you just heard was the first microphone recording, here's the second recording.

<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/2.mp3" controls="true">
Your browser does not support the audio element.
</audio>

Uno (one), dos (two), tres (three), cuatro (four), cinco (five), seis (six), siete (seven), ocho (eight), nueve (nine) y diez (ten). So we can do, is take these two microphone recorders and give them to an Unsupervised Learning algorithm called the cocktail party algorithm, and tell the algorithm - find structure in this data for you. And what the algorithm will do is listen to these audio recordings and say, you know it sounds like the two audio recordings are being added together or that have being summed together to produce these recordings that we had. Moreover, what the cocktail party algorithm will do is separate out these two audio sources that were being added or being summed together to form other recordings and, in fact, here's the first output of the cocktail party algorithm. 

<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/3.mp3" controls="true">
</audio>

One, two, three, four, five, six, seven, eight, nine, ten. So, I separated out the English voice in one of the recordings. And here's the second of it.

<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/4.mp3" controls="true">
Your browser does not support the audio element.
</audio>

Uno, dos, tres, quatro, cinco, seis, siete, ocho, nueve y diez. Not too bad, to give you one more example, here's another recording of another similar situation, here's the first microphone :  

<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/5.mp3" controls="true">
Your browser does not support the audio element.
</audio>

One, two, three, four, five, six, seven, eight, nine, ten. OK so the poor guy's gone home from the cocktail party and he 's now sitting in a room by himself talking to his radio. Here's the second microphone recording.
<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/6.mp3" controls="true">
Your browser does not support the audio element.
</audio>


One, two, three, four, five, six, seven, eight, nine, ten. When you give these two microphone recordings to the same algorithm, what it does, is again say, you know, it sounds like there are two audio sources, and moreover, the album says, here is the first of the audio sources I found.
<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/7.mp3" controls="true">
Your browser does not support the audio element.
</audio>

One, two, three, four, five, six, seven, eight, nine, ten. So that wasn't perfect, it got the voice, but it also got a little bit of the music in there. Then here's the second output to the algorithm. 
<audio src="http://q6gm8fomw.bkt.clouddn.com/gitpage/ml-andrew-ng/01/8.mp3" controls="true">
Your browser does not support the audio element.
</audio>

Not too bad, in that second output it managed to get rid of the voice entirely. And just, you know, cleaned up the music, got rid of the counting from one to ten. So you might look at an Unsupervised Learning algorithm like this.
**Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback based on the prediction results.**

### Summary

1. **Clustering**: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on. 
2. **Non-clustering**: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party ). 




## References

1. [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome)

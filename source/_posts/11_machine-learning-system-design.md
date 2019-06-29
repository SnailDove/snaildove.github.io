---
title: 11_machine-learning-system-design note11
date: 2018-01-11
copyright: true
categories: english
tags: [Machine Learning]
mathjax: false
mathjax2: false
---

In the next few videos I'd like to talk about **machine learning system design**. These videos will touch on **the main issues that you may face when designing a complex machine learning system, and will actually try to give advice on how to strategize putting together a complex machine learning system.** 

In case this next set of videos seems a little disjointed that's because these videos will touch on a range of the different issues that you may come across when designing complex learning systems. And ***even though the next set of videos may seem somewhat less mathematical, I think that this material may turn out to be very useful, and potentially huge time savers when you're building big machine learning systems.*** 

# Note

This personal note is written after studying the opening course on [the coursera website](https://www.coursera.org), [Machine Learning by Andrew NG](https://www.coursera.org/learn/machine-learning/home/welcome) . And images, audios of this note all comes from the opening course. 

# 01_building-a-spam-classifier

## Prioritizing What to Work On

Concretely, I'd like to begin with the issue of prioritizing how to spend your time on what to work on, and I'll begin with an example on spam classification. 

Let's say you want to build **a spam classifier**. Here are a couple of examples of obvious spam and non-spam emails. if the one on the left tried to sell things. And notice how spammers will deliberately misspell words, like Vincent with a 1 there, and mortgages. And on the right as maybe an obvious example of non-stamp email, actually email from my younger brother. Let's say we have a labeled training set of some number of spam emails and some non-spam emails denoted with labels y equals 1 or 0, how do we build a classifier using supervised learning to distinguish between spam and non-spam? 

**In order to apply supervised learning, the first decision we must make is how do we want to represent x, that is the features of the email.** Given the features x and the labels y in our training set, we can then train a classifier, for example using logistic regression. Here's one way to choose a set of features for our emails. We could come up with, say, a list of maybe a hundred words that we think are indicative of whether e-mail is spam or non-spam, for example, if a piece of e-mail contains the word 'deal' maybe it's more likely to be spam if it contains the word  'buy' maybe more likely to be spam, a word like 'discount' is more likely to be spam, whereas if a piece of email contains my name, Andrew, maybe that means the person actually knows who I am and that might mean it's less likely to be spam. And maybe for some reason I think the word "now" may be indicative of non-spam because I get a lot of urgent emails, and so on, and maybe we choose a hundred words or so. **Given a piece of email, we can then take this piece of email and encode it into a feature vector as follows.** I'm going to take my list of a hundred words and sort them in alphabetical order say. It doesn't have to be sorted. But, you know, here's a, here's my list of words, just count and so on, until eventually I'll get down to now, and so on and given a piece of e-mail like that shown on the right, I'm going to check and see whether or not each of these words appears in the e-mail and then I'm going to define a feature vector x where in this piece of an email on the right, my name doesn't appear so I'm gonna put a zero there. The word "by" does appear, so I'm gonna put a one there and I'm just gonna put one's or zeroes. I'm gonna put a one even though the word "by" occurs twice. I'm not gonna recount how many times the word occurs. The word "due" appears, I put a one there. The word "discount" doesn't appear, at least not in this this little short email, and so on. The word "now" does appear and so on. **So I put ones and zeroes in this feature vector depending on whether or not a particular word appears.** 

![example_of_spam-email_and_non-spam-email](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/1.png)

**And in this example my feature vector would have to mention one hundred, if I have a hundred, if if I chose a hundred words to use for this representation and each of my features $X_j$ will basically be $1$ if you have a particular word that, we'll call this word j, appears in the email and $X_j$ would be $0$ otherwise.**

Okay. So that gives me a feature representation of a piece of email. By the way, even though I've described this process as manually picking a hundred words, in practice what's most commonly done is to look through a training set, and in the training set depict the most frequently occurring n words where n is usually between ten thousand and fifty thousand, and use those as your features. So rather than manually picking a hundred words, here you look through the training examples and pick the most frequently occurring words like ten thousand to fifty thousand words, and those form the features that you are going to use to represent your email for spam classification. 

**Now, if you're building a spam classifier, one question that you may face is, what's the best use of your time in order to make your spam classifier have higher accuracy, you have lower error.** One natural inclination is going to collect lots of data. Right? And in fact there's this tendency to think that, well the more data we have the better the algorithm will do. And in fact, in the email spam domain, there are actually pretty serious projects called Honey Pot Projects, which create fake email addresses and try to get these fake email addresses into the hands of spammers and use that to try to collect tons of spam email, and therefore you know, get a lot of spam data to train learning algorithms. But we've already seen in the previous sets of videos that getting lots of data will often help, but not all the time. But for most machine learning problems, there are a lot of other things you could usually imagine doing to improve performance. For spam, one thing you might think of is to develop more sophisticated features on the email, maybe based on the email routing information. And this would be information contained in the email header. So, when spammers send email, very often they will try to obscure the origins of the email, and maybe use fake email headers. Or send email through very unusual sets of computer service. Through very unusual routes, in order to get the spam to you. And some of this information will be reflected in the email header. And so one can imagine, looking at the email headers and trying to develop more sophisticated features to capture this sort of email routing information to identify if something is spam. Something else you might consider doing is to look at the email message body, that is the email text, and try to develop more sophisticated features. For example, should the word 'discount' and the word 'discounts' be treated as the same words or should we have treat the words 'deal' and 'dealer' as the same word? Maybe even though one is lower case and one in capitalized in this example. Or do we want more complex features about punctuation because maybe spam is using exclamation marks a lot more. I don't know. And along the same lines, maybe we also want to develop more sophisticated algorithms to detect and maybe to correct to deliberate misspellings, like mortgage, medicine, watches. Because spammers actually do this, because if you have watches with a 4 in there then well, with the simple technique that we talked about just now, the spam classifier might not equate this as the same thing as the word "watches," and so it may have a harder time realizing that something is spam with these deliberate misspellings. And this is why spammers do it. While working on a machine learning problem, very often you can brainstorm lists of different things to try, like these. By the way, I've actually worked on the spam problem myself for a while. And I actually spent quite some time on it. And even though I kind of understand the spam problem, I actually know a bit about it, I would actually have a very hard time telling you of these four options which is the best use of your time so what happens, frankly what happens far too often is that a research group or product group will randomly fixate on one of these options. And sometimes that turns out not to be the most fruitful way to spend your time depending, you know, on which of these options someone ends up randomly fixating on. By the way, in fact, if you even get to the stage where you brainstorm a list of different options to try, you're probably already ahead of the curve. Sadly, what most people do is instead of trying to list out the options of things you might try, what far too many people do is wake up one morning and, for some reason, just, you know, have a weird gut feeling that, "Oh let's have a huge honeypot project to go and collect tons more data" and for whatever strange reason just sort of wake up one morning and randomly fixate on one thing and just work on that for six months. But I think we can do better. And in particular what I'd like to do in the next video is tell you about the concept of error analysis and talk about the way where you can try to have a more systematic way to choose amongst the options of the many different things you might work, and therefore be more likely to select what is actually a good way to spend your time, you know for the next few weeks, or next few days or the next few months.

**System Design Example:**   

Given a data set of emails, we could construct a vector for each email. Each entry in this vector represents a word. The vector normally contains 10,000 to 50,000 entries gathered by finding the most frequently used words in our data set.  If a word is to be found in the email, we would assign its respective entry a 1, else if it is not found, that entry would be a 0. Once we have all our x vectors ready, we train our algorithm and finally, we could use it to classify if an email is a spam or not. 

![Building_a_spam_classifier](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/2.png)

So how could you spend your time to improve the accuracy of this classifier? 

- Collect lots of data (for example "honeypot" project but doesn't always work)   
- Develop sophisticated features (for example: using email header data in spam emails)   
- Develop sophisticated features for message body (for example: should“discount” and “discounts” be treated as the same word? How about “deal” and “Dealer”? Features about punctuation)? 
- Develop algorithms to process your input in different ways (recognizing misspellings in spam, for example, med1cine, m0rtgage, w4tches).   

  It is difficult to tell which of the options will be most helpful.

## Error Analysis

The recommended approach to solving machine learning problems is to:  

- ***Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.***   
- ***Plot learning curves to decide if more data, more features, etc. are likely to help.***   
- ***Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.***   

For example, assume that we have 500 emails and our algorithm misclassifies a 100 of them. We could manually analyze the 100 emails and categorize them based on what type of emails they are. We could then try to come up with new cues and features that would help us classify these 100 emails correctly. Hence, if most of our misclassified emails are those which try to steal passwords, then we could find some features that are particular to those emails and add them to our model. 

![error_analysis](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/3.png)

We could also see how classifying each word according to its root changes our error rate:

![The_importance_of_numerical_evaluation](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/4.png)

It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance. For example if we use stemming, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), and get a 3% error rate instead of 5%, then we should definitely add it to our model. However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature.  Hence, we should try new things, get a numerical value for our error rate, and based on our result decide whether we want to keep the new feature or not.

# 02_handling-skewed-data

## 01_error-metrics-for-skewed-classes

In the previous video, I talked about error analysis and the importance of having error metrics, that is of having a single real number evaluation metric for your learning algorithm to tell how well it's doing. 

In the context of evaluation and of error metrics, there is one important case, where it's particularly tricky to come up with an appropriate error metric, or evaluation metric, for your learning algorithm. That case is the case of what's called **skewed classes**. Let me tell you what that means. 

Consider the problem of cancer classification, where we have features of medical patients and we want to decide whether or not they have cancer. So this is like the malignant versus benign tumor classification example that we had earlier. So let's say y equals 1 if the patient has cancer and y equals 0 if they do not. We have trained the progression classifier and let's say we test our classifier on a test set and find that we get 1 percent error. So, we're making 99% correct diagnosis. Seems like a really impressive result, right. We're correct 99% percent of the time. 

But now, let's say we find out that only 0.5 percent of patients in our training test sets actually have cancer. So only half a percent of the patients that come through our screening process have cancer. In this case, the 1% error no longer looks so impressive.

![cancer_classification_example_of_skewed-class](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/5.png)

And in particular, here's a piece of code, here's actually a piece of non learning code that takes this input of features x and it ignores it. ***It just sets y equals 0 and always predicts, you know, nobody has cancer and this algorithm would actually get 0.5 percent error. So this is even better than the 1% error that we were getting just now and this is a non learning algorithm that you know, it is just predicting y equals 0 all the time. So this setting of when the ratio of positive to negative examples is very close to one of two extremes, where, in this case, the number of positive examples is much, much smaller than the number of negative examples because y equals one so rarely, this is what we call the case of skewed classes.*** 

We just have a lot more of examples from one class than from the other class. And by just predicting y equals 0 all the time, or maybe our predicting y equals 1 all the time, an algorithm can do pretty well. So the problem with using classification error or classification accuracy as our evaluation metric is the following. 

Let's say you have one joining algorithm that's getting 99.2% accuracy. So, that's a 0.8% error. Let's say you make a change to your algorithm and you now are getting 99.5% accuracy. That is 0.5% error. So, is this an improvement to the algorithm or not? One of the nice things about having a single real number evaluation metric is this helps us to quickly decide if we just need a good change to the algorithm or not. **By going from 99.2% accuracy to 99.5% accuracy. You know, did we just do something useful or did we just replace our code with something that just predicts y equals zero more often?  So, if you have very skewed classes it becomes much harder to use just classification accuracy, because you can get very high classification accuracies or very low errors, and it's not always clear if doing so is really improving the quality of your classifier because predicting y equals 0 all the time doesn't seem like a particularly good classifier.** But just predicting y equals 0 more often can bring your error down to, you know, maybe as low as 0.5%. 

When we're faced with such a skewed classes therefore we would want to come up with a different error metric or a different evaluation metric. One such evaluation metric are what's called **precision and recall**. Let me explain what that is. Let's say we are evaluating a classifier on the test set. For the examples in the test set the actual class of that example in the test set is going to be either one or zero, right, if there is a binary classification problem. And what our learning algorithm will do is it will, you know, predict some value for the class and our learning algorithm will predict the value for each example in my test set and the predicted value will also be either one or zero. So let me draw a two by two table as follows, depending on a full of these entries depending on what was the actual class and what was the predicted class. If we have an example where the actual class is one and the predicted class is one then that's called an example that's a ***true positive***, meaning our algorithm predicted that it's positive and in reality the example is positive. If our learning algorithm predicted that something is negative, class zero, and the actual class is also class zero then that's what's called a ***true negative***. We predicted zero and it actually is zero. To find the other two boxes, if our learning algorithm predicts that the class is one but the actual class is zero, then that's called a ***false positive***. So that means our algorithm for the patient is cancelled out in reality if the patient does not. And finally, the last box is a zero, one. That's called a ***false negative*** because our algorithm predicted zero, but the actual class was one. And so, we have this little sort of two by two table based on what was the actual class and what was the predicted class. So here's a different way of evaluating the performance of our algorithm. We're going to compute two numbers. 

![definitions_of_precision_and_recall](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/6.png)

The first is called ***precision*** - and what that says is, of all the patients where we've predicted that they have cancer, what fraction of them actually have cancer? So let me write this down, the precision of a classifier is the number of true positives divided by the number that we predicted as positive, right? So of all the patients that we went to those patients and we told them, "We think you have cancer." Of all those patients, what fraction of them actually have cancer? So that's called precision. **And another way to write this would be true positives and then in the denominator is the number of predicted positives, and so that would be the sum of the, you know, entries in this first row of the table. So it would be true positives divided by positives. I'm going to abbreviate positive as POS and then plus false positives, again abbreviating positive using POS.** So that's called precision, and as you can tell high precision would be good. That means that all the patients that we went to and we said, "You know, we're very sorry. We think you have cancer," ***high precision*** means that of that group of patients most of them we had actually made accurate predictions on them and they do have cancer. 

The second number we're going to compute is called ***recall***, and what recall say is, if all the patients in, let's say, in the test set or the cross-validation set, but if all the patients in the data set that actually have cancer, what fraction of them that we correctly detect as having cancer. So if all the patients have cancer, how many of them did we actually go to them and you know, correctly told them that we think they need treatment. **So, writing this down, recall is defined as the number of positives, the number of true positives, meaning the number of people that have cancer and that we correctly predicted have cancer and we take that and divide that by, divide that by the number of actual positives, so this is the right number of actual positives of all the people that do have cancer.** What fraction do we directly flag and you know, send the treatment. So, to rewrite this in a different form, the denominator would be the number of actual positives as you know, is the sum of the entries in this first column over here. And so writing things out differently, this is therefore, the number of true positives, divided by the number of true positives plus the number of false negatives. And so once again, having a high recall would be a good thing. 

So by computing precision and recall this will usually give us a better sense of how well our classifier is doing. And in particular if we have a learning algorithm that predicts y equals zero all the time, if it predicts no one has cancer, then this classifier will have a recall equal to zero, because there won't be any true positives and so that's a quick way for us to recognize that, you know, a classifier that predicts y equals 0 all the time, just isn't a very good classifier. And more generally, even for settings where we have very skewed classes, it's not possible for an algorithm to sort of "cheat" and somehow get a very high precision and a very high recall by doing some simple thing like predicting y equals 0 all the time or predicting y equals 1 all the time. **And so we're much more sure that a classifier of a high precision or high recall actually is a good classifier, and this gives us a more useful evaluation metric that is a more direct way to actually understand whether, you know, our algorithm may be doing well.** So one final note in the definition of precision and recall, that we would define precision and recall, usually we use the convention that y is equal to 1, in the presence of the more rare class. So if we are trying to detect. rare conditions such as cancer, hopefully that's a rare condition, precision and recall are defined setting y equals 1, rather than y equals 0, to be sort of that the presence of that rare class that we're trying to detect. And by using precision and recall, we find, what happens is that even if we have very skewed classes, it's not possible for an algorithm to you know, "cheat" and predict y equals 1 all the time, or predict y equals 0 all the time, and get high precision and recall. And in particular, if a classifier is getting high precision and high recall, then we are actually confident that the algorithm has to be doing well, even if we have very skewed classes. ***So for the problem of skewed classes precision recall gives us more direct insight into how the learning algorithm is doing and this is often a much better way to evaluate our learning algorithms, than looking at classification error or classification accuracy, when the classes are very skewed.***

## 02_trading-off-precision-and-recall

In the last video,we talked about precision and recall as an evaluation metric for classification problems with skewed constants. **For many applications, we'll want to somehow control the trade-off between precision and recall.** Let me tell you how to do that and also show you some even more effective ways to use precision and recall as an evaluation metric for learning algorithms.

As a reminder,here are the definitions of precision and recall from the previous video. Let's continue our cancer classification example, where y equals 1 if the patient has cancer and y equals 0 otherwise. And let's say we're trained in logistic regression classifier which outputs probability between 0 and 1. So, as usual, we're going to predict 1, y equals 1, if h(x) is greater or equal to 0.5. And predict 0 if the hypothesis outputs a value less than 0.5. And this classifier may give us some value for precision and some value for recall. But now, suppose we want to predict that the patient has cancer only if we're very confident that they really do. Because if you go to a patient and you tell them that they have cancer, it's going to give them a huge shock. What we give is a seriously bad news, and they may end up going through a pretty painful treatment process and so on. And so maybe we want to tell someone that we think they have cancer only if they are very confident. One way to do this would be to modify the algorithm, so that instead of setting this threshold at 0.5, we might instead say that we will predict that y is equal to 1 only if h(x) is greater or equal to 0.7. So this is like saying, we'll tell someone they have cancer only if we think there's a greater than or equal to, 70% chance that they have cancer. And, if you do this, then you're predicting someone has cancer
only when you're more confident and so you end up with a classifier that has higher precision. Because all of the patients that you're going to and saying, we think you have cancer, although those patients are now ones that you're pretty confident actually have cancer. And **so a higher fraction of the patients that you predict have cancer will actually turn out to have cancer because making those predictions only if we're pretty confident. But in contrast this classifier will have lower recall because now we're going to make predictions, we're going to predict y = 1 on a smaller number of patients.** Now, can even take this further. Instead of setting the threshold at 0.7, we can set this at 0.9. Now we'll predict y=1 only if we are more than 90% certain that the patient has cancer. And so, a large fraction of those patients will turn out to have cancer. And so this would be a higher precision classifier will have lower recall because we want to correctly detect that those patients have cancer. **Now consider a different example. Suppose we want to avoid missing too many actual cases of cancer, so we want to avoid false negatives. In particular, if a patient actually has cancer, but we fail to tell them that they have cancer then that can be really bad. Because if we tell a patient that they don't have cancer, then they're not going to go for treatment. And if it turns out that they have cancer, but we fail to tell them they have cancer, well, they may not get treated at all. And so that would be a really bad outcome because they die because we told them that they don't have cancer. They fail to get treated, but it turns out they actually have cancer.** So, suppose that, when in doubt, we want to predict that y=1. So, when in doubt, we want to predict that they have cancer so that at least they look further into it, and these can get treated in case they do turn out to have cancer. **In this case, rather than setting higher probability threshold, we might instead take this value and instead set it to a lower value.** So maybe 0.3 like so, right? And by doing so, we're saying that,you know what, if we think there's more than a 30% chance that they have cancer we better be more conservative and tell them that they may have cancer so that they can seek treatment if necessary. **And in this case what we would have is going to be a higher recall classifier, because we're going to be correctly flagging a higher fraction of all of the patients that actually do have cancer.** But we're going to end up with lower precision because a higher fraction of the patients that we said have cancer, a high fraction of them will turnout not to have cancer after all. And by the way, just as a sider, when I talk about this to other students, I've been told before, it's pretty amazing, some of my students say, is how I can tell the story both ways. **Why we might want to have higher precision or higher recall and the story actually seems to work both ways. But I hope the details of the algorithm is true and the more general principle is depending on where you want, whether you want higher precision- lower recall, or higher recall- lower precision.** 

![trading-off_precision_and_recall](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/7.png)

You can end up predicting y=1 when h(x) is greater than some threshold. And so in general, for most classifiers there is going to be a trade off between precision and recall, and as you vary the value of this threshold that we join here, you can actually plot out some curve that trades off precision and recall. Where a value up here, this would correspond to a very high value of the threshold, maybe threshold equals 0.99. So that's saying, predict y=1 only if we're more than 99% confident, at least 99% probability this one. So that would be a high precision, relatively low recall. Where as the point down here, will correspond to a value of the threshold that's much lower, maybe equal 0.01, meaning, when in doubt at all, predict y=1, and if you do that, you end up with a much lower precision, higher recall classifier. And as you vary the threshold, if you want you can actually trace of a curve for your classifier to see the range of different values you can get for precision recall. And by the way, the precision-recall curve can look like many different shapes. Sometimes it will look like this, sometimes it will look like that. Now there are many different possible shapes for the precision-recall curve, depending on the details of the classifier.

**So, this raises another interesting question which is, is there a way to choose this threshold automatically? Or more generally, if we have a few different algorithms or a few different ideas for algorithms, how do we compare different precision recall numbers? Concretely, suppose we have three different learning algorithms. So actually, maybe these are three different learning algorithms, maybe these are the same algorithm but just with different values for the threshold. How do we decide which of these algorithms is best?**

![F1_score](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/8.png)

# 03_using-large-data-sets

In the previous video, we talked about evaluation metrics. In this video, I'd like to switch tracks a bit and touch on another important aspect of machine learning system design, which will often come up, which is **the issue of how much data to train on**.  

Now, in some earlier videos, ***I had cautioned against blindly going out and just spending lots of time collecting lots of data, because it's only sometimes that that would actually help.*** But it turns out that under certain conditions, and I will say in this video what those conditions are, ***getting a lot of data and training on a certain type of learning algorithm, can be a very effective way to get a learning algorithm to do very good performance. And this arises often enough that if those conditions hold true for your problem and if you're able to get a lot of data, this could be a very good way to get a very high performance learning algorithm.*** So in this video, let's talk more about that. Let me start with a story. 

## Michelle Banko and Eric Broule

Many, many years ago, two researchers that I know, Michelle Banko and Eric Broule ran the following fascinating study. They were interested in studying the effect of using different learning algorithms versus trying them out on different training set sciences, they were considering the problem of classifying between confusable words, so for example, in the sentence: for breakfast I ate, should it be to, two or too? Well, for this example, for breakfast I ate two, 2 eggs. So, this is one example of a set of confusable words and that's a different set. So they took machine learning problems like these, sort of supervised learning problems to try to categorize what is the appropriate word to go into a certain position in an English sentence. 

![img](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/9.png) 

They took a few different learning algorithms which were, you know, sort of considered state of the art back in the day, when they ran the study in 2001, so they took a variance, roughly a variance on logistic regression called the Perceptron. They also took some of their algorithms that were fairly out back then but somewhat less used now so when the algorithm also very similar to which is a regression but different in some ways, much used somewhat less, used not too much right now took what's called a memory based learning algorithm again used somewhat less now. But I'll talk a little bit about that later. And they used a naive based algorithm, which is something they'll actually talk about in this course. The exact algorithms of these details aren't important. Think of this as, you know, just picking four different classification algorithms and really the exact algorithms aren't important. **But what they did was they varied the training set size and tried out these learning algorithms on the range of training set sizes and that's the result they got.**

![img](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/10.png) 

And the trends are very clear, right? 

1. first, most of these algorithms give remarkably similar performance. 
2. And second, as the training set size increases, on the horizontal axis is the training set size in millions go from, you know, a hundred thousand up to a thousand million that is a billion training examples. The performance of the algorithms all pretty much monotonically increase and the fact that if you pick any algorithm, may be pick a "inferior algorithm", but if you give that "inferior algorithm" more data, then from these examples, it looks like it will most likely beat even a "superior algorithm". 

So since this original study which is very influential, **there's been a range of many different studies showing similar results that show that many different learning algorithms you know tend to, can sometimes, depending on details, can give pretty similar ranges of performance, but what can really drive performance is you can give the algorithm a ton of training data.** And this is, results like these has led to a saying in machine learning that often in machine learning **it's not who has the best algorithm that wins, it's who has the most data So when is this true and when is this not true?** Because we have a learning algorithm for which this is true then getting a lot of data is often maybe the best way to ensure that we have an algorithm with very high performance rather than you know, debating worrying about exactly which of these items to use.

## the features x have sufficient information

Let's try to lay out a set of assumptions under which having a massive training set we think will be able to help. Let's assume that in our machine learning problem, the features x have sufficient information, with which we can use to predict y accurately. 

For example, if we take the confusable words all of them that we had on the previous slide. Let's say that it features x capture what are the surrounding words around the blank that we're trying to fill in. So the features capture then we want to have, sometimes for breakfast I have black eggs. Then yeah that is pretty much information to tell me that the word I want in the middle is TWO and that is not word TO and its not the word TOO. **So the features capture, you know, one of these surrounding words then that gives me enough information to pretty unambiguously decide what is the label y or in other words what is the word that I should be using to fill in that blank out of this set of three confusable words.** So that's an example what the futures x has sufficient information for specific y. 

For a counter example. **Consider a problem of predicting the price of a house from only the size of the house and from no other features.** So if you imagine I tell you that a house is, you know, 500 square feet but I don't give you any other features. I don't tell you that the house is in an expensive part of the city. Or if I don't tell you that the house, the number of rooms in the house, or how nicely furnished the house is, or whether the house is new or old. If I don't tell you anything other than that this is a 500 square foot house, well there's so many other factors that would affect the price of a house other than just the size of a house that if all you know is the size, it's actually very difficult to predict the price accurately. So that would be a counter example to this assumption that the features have sufficient information to predict the price to the desired level of accuracy. 

## domain knowledge

The way I think about testing this assumption, one way I often think about it is, how often I ask myself. Given the input features x, given the features, given the same information available as well as learning algorithm. If we were to go to **human expert in this domain**. Can a human experts actually or can human expert confidently predict the value of y. ***For this first example if we go to, you know an expert human English speaker.*** You go to someone that speaks English well, right, then a human expert in English just read most people like you and me will probably we would probably be able to predict what word should go in here, to a good English speaker can predict this well, and so this gives me confidence that x allows us to predict y accurately, but in contrast if we go to ***an expert in house prices.*** Like maybe an expert realtor, right, someone who sells houses for a living. If I just tell them the size of a house and I tell them what the price is well even an expert in pricing or selling houses wouldn't be able to tell me and so this is fine that for the housing price example knowing only the size doesn't give me enough information to predict the price of the house. So, let's say, this assumption holds. Let's see then, when having a lot of data could help. Suppose the features have enough information to predict the value of y. 

And let's suppose we use a learning algorithm with a large number of parameters so maybe logistic regression or linear regression with a large number of features. Or one thing that I sometimes do, one thing that I often do actually is using neural network with many hidden units. That would be another learning algorithm with a lot of parameters. So these are all powerful learning algorithms with a lot of parameters that can fit very complex functions. So, I'm going to call these, I'm going to think of these as low-bias algorithms because you know we can fit very complex functions and because we have a very powerful learning algorithm, they can fit very complex functions. Chances are, if we run these algorithms on the data sets, it will be able to fit the training set well, and so hopefully the training error will be slow. 

## Large data rationale

![img](http://pt8q6wt5q.bkt.clouddn.com/gitpage/ml-andrew-ng/11/11.png) 

**Now let's say, we use a massive, massive training set, in that case, if we have a huge training set, then hopefully even though we have a lot of parameters but if the training set is sort of even much larger than the number of parameters then hopefully these albums will be unlikely to overfit.** Right, **because we have such a massive training set and by unlikely to overfit what that means is that the training error will hopefully be close to the test error.** Finally putting these two together that the train set error is small and the test set error is close to the training error what this two together imply is that hopefully the test set error will also be small. 

**Another way to think about this is that in order to have a high performance learning algorithm we want it not to have high bias and not to have high variance. So the bias problem we're going to address by making sure we have a learning algorithm with many parameters and so that gives us a low bias algorithm and by using a very large training set, this ensures that we don't have a variance problem here.** So hopefully our algorithm will have no variance and so is by pulling these two together, that we end up with a low bias and a low variance learning algorithm and this allows us to do well on the test set. And fundamentally it's a key ingredients of assuming that the features have enough information and we have a rich class of functions that's why it guarantees low bias, and then it having a massive training set that that's what guarantees more variance. So this gives us a set of conditions rather hopefully some understanding of what's the sort of problem where if you have a lot of data and you train a learning algorithm with lot of parameters, that might be a good way to give a high performance learning algorithm 

## The Key Test

and really, I think **the key test** that I often ask myself are **first**, can a human experts look at the features x and confidently predict the value of y. Because that's sort of a certification that y can be predicted accurately from the features x and **second**, can we actually get a large training set, and train the learning algorithm with a lot of parameters in the training set and if you can't do both then that's more often give you a very kind performance learning algorithm.

# summary

## Prioritizing What to Work On 

Different ways we can approach a machine learning problem: 

* Collect lots of data (for example "honeypot" project but doesn't always work) 
* Develop sophisticated features (for example: using email header data in spam emails) 
* Develop algorithms to process your input in different ways (recognizing misspellings in spam). 

It is difficult to tell which of the options will be helpful. 

## Error Analysis 

The recommended approach to solving machine learning problems is: 

* Start with a simple algorithm, implement it quickly, and test it early. 
* Plot learning curves to decide if more data, more features, etc. will help 
* Error analysis: manually examine the errors on examples in the cross validation set and try to spot a trend. 

It's important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance. 
You may need to process your input before it is useful. For example, if your input is a set of words, you may want to treat the same word with different forms (fail/failing/failed) as one word, so must use "stemming software" to recognize them all as one.

## Error Metrics for Skewed Classes 

It is sometimes difficult to tell whether a reduction in error is actually an improvement of the algorithm. 

For example: In predicting a cancer diagnoses where 0.5% of the examples have cancer, we find our learning algorithm has a 1% error. However, if we were to simply classify every single example as a 0, then our error would reduce to 0.5% even though we did not improve the algorithm. 

This usually happens with **skewed classes** ; that is, when our class is very rare in the entire data set. 
Or to say it another way, when we have lot more examples from one class than from the other class. 
For this we can use **Precision/Recall** . 

* Predicted: 1, Actual: 1 --- True positive 
* Predicted: 0, Actual: 0 --- True negative 
* Predicted: 0, Actual, 1 --- False negative 
* Predicted: 1, Actual: 0 --- False positive 

**Precision** : of all patients we predicted where y=1, what fraction actually has cancer? 
$$\dfrac{\text{True Positives}}{\text{Total number of predicted positives}} = \dfrac{\text{True Positives}}{\text{True Positives}+\text{False positives}}$$ 
**Recall** : Of all the patients that actually have cancer, what fraction did we correctly detect as having cancer? 
$$\dfrac{\text{True Positives}}{\text{Total number of actual positives}}= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False negatives}}$$ 
These two metrics give us a better sense of how our classifier is doing. We want both precision and recall to be high. 
In the example at the beginning of the section, if we classify all patients as 0, then our **recall** will be $\dfrac{0}{0 + f} = 0$, so despite having a lower error percentage, we can quickly see it has worse recall. 
Accuracy = $\frac {true\ positive + true\ negative} {total\ population}$ 
Note 1: if an algorithm predicts only negatives like it does in one of exercises, the precision is not defined, it is impossible to divide by 0. F1 score will not be defined too. 

## Trading Off Precision and Recall 

We might want a **confident prediction** of two classes using logistic regression. One way is to increase our threshold: 
Predict 1 if: $h_\theta(x) \geq 0.7$ 
Predict 0 if: $h_\theta(x) < 0.7$
This way, we only predict cancer if the patient has a 70% chance. 
Doing this, we will have **higher precision** but **but lower recall**(refer to the definitions in the previous section). 
In the opposite example, we can lower our threshold: 
Predict 1 if: $h_\theta(x) \geq 0.3$ 
Predict 0 if: $h_\theta(x) < 0.3$ 
That way, we get a very **safe prediction**. This will cause higher recall but lower precision . 
The greater the threshold, the greater the precision and the lower the recall. 
The lower the threshold, the greater the recall and the lower the precision. 
In order to turn these two metrics into one single number, we can take the **F value** . 
One way is to take the **average** : 
$$\dfrac{P+R}{2}$$ 
This does not work well. If we predict all y=0 then that will bring the average up despite having 0 recall. If we predict all examples as y=1, then the very high recall will bring up the average despite having 0 precision. 
A better way is to compute the F Score (or **F1 score**): 
$$\text{F Score} = 2\dfrac{PR}{P + R}$$ 
In order for the F Score to be large, both precision and recall must be large. 
We want to train precision and recall on **the cross validation** set so as not to bias our test set. 
## Data for Machine Learning 
How much data should we train on? 
In certain cases, an "inferior algorithm," if given enough data, can outperform a superior algorithm with less data. 
We must choose our features to have **enough** information. A useful test is: Given input x, would a human expert be able to confidently predict y? 
**Rationale for large data** : if we have a low bias algorithm (many features or hidden units making a very complex function), then the larger the training set we use, the less we will have overfitting (and the more accurate the algorithm will be on the test set). 
## Quiz instructions 
When the quiz instructions tell you to enter a value to "two decimal digits", what it really means is "two significant digits". So, just for example, the value 0.0123 should be entered as "0.012", not "0.01". 
References: 
https://class.coursera.org/ml/lecture/index 
http://www.cedar.buffalo.edu/~srihari/CSE555/Chap9.Part2.pdf 
http://blog.stephenpurpura.com/post/13052575854/managing-bias-variance-tradeoff-in-machine-learning 
http://www.cedar.buffalo.edu/~srihari/CSE574/Chap3/Bias-Variance.pdf

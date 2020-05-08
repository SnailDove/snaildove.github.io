---
title: PCA in action
mathjax: true
mathjax2: true
categories: English
tags: [Machine Learning, Python Data Science Cookbook]
date: 2017-02-11
comments: true
copyright: true
toc: true
top: true
---

This is my personal learning note of the book,  *[Python Data Science Cookook](https://www.amazon.com/Python-Data-Science-Cookbook-Subramanian/dp/1784396400)*.

## Tip

before learning about the following example , we need to have the notion of the principle of PCA

1. the principle of PCA in English refers to https://en.wikipedia.org/wiki/Principal_component_analysis
2. the principle of PCA in Chinese refers to http://blog.codinglabs.org/articles/pca-tutorial.html

this example of data set reder to https://archive.ics.uci.edu/ml/datasets/Iris

Let’s use the Iris dataset to understand how to use PCA efficiently in reducing the dimension of the dataset. The Iris dataset contains measurements for 150 iris flowers from three different species. The three classes in the Iris dataset are as follows:

1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

The following are the four features in the Iris dataset:

1. The sepal length in cm
2. The sepal width in cm
3. The petal length in cm
4. The petal width in cm

Can we use, say, two columns instead of all the four columns to express most of the variations in the data ?Can we reduce the number of columns from four to two and still achieve a good accuracy for our classifier?

## The steps of PCA algorithm

if you have the notion of the principle of PCA , the following steps is easy to understand :

1. Standardize the dataset to have a zero mean value.
2. Find the correlation matrix for the dataset and unit standard deviation value.
3. Reduce the Correlation matrix matrix into its Eigenvectors and values.
4. Select the top nEigenvectors based on the Eigenvalues sorted in descending order.
5. Project the input Eigenvectors matrix into the new subspace.

```python2
#!/usr/bin/env python2

# -*- coding: utf-8 -*-

"""
@author: snaidove
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
import scipy
import matplotlib.pyplot as plt

# Load Iris data
data = load_iris()
x = data['data']
y = data['target']

# Since PCA is an unsupervised method, we will not be using the target variable y
#scale the data matrix x to have zero mean and unit standard deviation. The rule of thumb is that if all your columns are measured in the same scale in your data and have the same unit of measurement, you don’t have to scale the data. This will allow PCA to capture these basic units with the maximum variation:
x_s = scale(x,with_mean=True, with_std=True, axis=0)
# Calculate correlation matrix
x_c = np.corrcoef(x_s.T)
# Find eigen value and eigen vector from correlation matrix
eig_val,r_eig_vec = scipy.linalg.eig(x_c)
print 'Eigen values \n%s'%(eig_val)
print '\n Eigen vectors \n%s'%(r_eig_vec)
#print "PCP': "
#print r_eig_vec.dot(x_c).dot(r_eig_vec.T);
# Select the first two eigen vectors.
w = r_eig_vec[:,0:2]
#Project the dataset in to the dimension from 4 dimension to 2 using the right eignen vector
x_rd = x_s.dot(w)
# Scatter plot the new two dimensions
plt.close('all');
plt.figure(1)
plt.scatter(x_rd[:,0],x_rd[:,1],marker='o',c=y)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
```

output

```
Eigen values 
[ 2.91081808+0.j  0.92122093+0.j  0.14735328+0.j  0.02060771+0.j]

 Eigen vectors 
[[ 0.52237162 -0.37231836 -0.72101681  0.26199559][-0.26335492 -0.92555649 0.24203288 -0.12413481]
 [ 0.58125401 -0.02109478  0.14089226 -0.80115427][ 0.56561105 -0.06541577 0.6338014 0.52354627]]
```

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/PythonDataScienceCookbook/output_0_2.png)

## how many components/dimensions should we choose?

The following are a 2 ways to select the components more empirically:

### The Eigenvalue criterion:

An Eigenvalue of one would mean that the component would explain about one variable’s worth of variability. So, according to this criterion, a component should at least explain one variable’s worth of variability. We can say that we will include only those Eigenvalues whose value is greater than or equal to one. Based on your data set you can set the threshold. In a very large dimensional dataset including components capable of explaining only one variable may not be very useful.

### The proportion of the variance explained criterion:

Let’s run the following code:
```python2
print "Component, Eigen Value, % of Variance, Cummulative %"
cum_per = 0
per_var = 0
for i,e_val in enumerate(eig_val):
  per_var = round((e_val / len(eig_val)),3)
  cum_per+=per_var
  print ('%d, %0.2f, %0.2f, %0.2f')%(i+1, e_val, per_var*100,cum_per*100)
```
The output is as follows:
```python2
Component, Eigen Value, % of Variance, Cummulative %
1, 2.91, 72.80, 72.80
2, 0.92, 23.00, 95.80
3, 0.15, 3.70, 99.50
4, 0.02, 0.50, 100.00
```
For each component, we printed **the Eigenvalue, percentage of the variance explained by that component, and cumulative percentage value of the variance explained.** For example, component 1 has an Eigenvalue of 2.91; 2.91/4 gives the percentage of the variance explained, which is 72.80%. **Now, if we include the first two components, then we can explain 95.80% of the variance(namely distribution) in the data.**
The decomposition of a correlation matrix into its Eigenvectors and values is a general technique that can be applied to any matrix. In this case, we will apply it to a correlation matrix in order to understand the principal axes of data distribution, that is, axes through which the maximum variation in the data is observed.

## A drawback of PCA

A drawback of PCA worth mentioning here is that **it is computationally expensive operation**. Finally a point about numpy’s corrcoeff function. **The corrcoeff function will standardize your data internally as a part of its calculation. But since we want to explicitly state the reason for scaling, we have included it in our recipe.**

## When would PCA work?

**The input dataset should have correlated columns for PCA to work effectively.** Without a correlation of the input variables, PCA cannot help us.

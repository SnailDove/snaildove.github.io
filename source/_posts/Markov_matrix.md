---
title: 马尔科夫矩阵和正矩阵
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-06 20:16:00
tags: [linear_algebra, 线性代数]
commets: true
toc: true
---

笔记源自：清华大学公开课：线性代数2——第9讲：马尔科夫矩阵和正矩阵



## 引言

![preface](http://img.blog.csdn.net/20180102233306825?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

马尔科夫链详细参考：[Markov chain](https://en.wikipedia.org/wiki/Markov_chain)

## Markov Matrix

### 正矩阵

![introduction_of_positive_matrix](http://img.blog.csdn.net/20180102233533038?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 马尔科夫矩阵定义

![defintion_of_Markov_matrix](http://img.blog.csdn.net/20180102233816802?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 马尔科夫矩阵性质

![properties_of_Markov_matrix](http://img.blog.csdn.net/20180102233938256?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 正马尔科夫矩阵

![positive_Markov_matrix](http://img.blog.csdn.net/20180102234234466?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 正马尔科夫矩阵的性质

![正马尔科夫矩阵的性质](http://img.blog.csdn.net/20180102234529456?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 例子

![example_of_positive_Markov_matrix](http://img.blog.csdn.net/20180102234932701?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 人口流动模型

![population_flow_model](http://img.blog.csdn.net/20180102235612652?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 正矩阵

![positive-matrix_and_spectral_radius](http://img.blog.csdn.net/20180102235700228?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

谱半径Spectral radius 定义为谱半径是矩阵特征值模的最大值，而非最大特征值，注意：矩阵来自于线性变换（也叫线性算子），因此线性变换也有谱半径，详询wiki: [谱半径Spectral radius](https://en.wikipedia.org/wiki/Spectral_radius)。

##Perron-Frobenius theorem

**这个原理应用在统计推断，经济，人口统计学，搜索引擎的基础。**

![Perron-Frobenius_theorem](http://img.blog.csdn.net/20180102235733277?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveW91MTMxNDUyMG1l/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 

---
title:  期望定义的由来 
mathjax: true
mathjax2: true
categories: 中文
tags: [probability, 概率论]
date: 2017-08-14 20:16:00
commets: true
toc: true
copyright: true
---
## 前言

我们很早就学到某个随机变量$X$的期望就是$X$的所有取值相对于它的概率的加权平均， 但是这是为什么呢？很多人都有疑问，后来看了MIT教授写的 [Introduction to Probability, 2nd Edition](http://www.athenasc.com/probbook.html) 书，豁然开朗，以此小计一篇。

## 例子

我们先以一个例子入手：假设你有机会转动一个幸运轮许多次，每次转动后幸运轮都会出现一个数字（数字即奖金数），不妨设为$m_i, i$表示第$i$次转动幸运轮，而且这些数字出现的概率分别为$p_i$，那么每次你期望得到的奖金数是多少呢？此处“每次”和”期望“都是一些不确定的词汇，我们来一一明确它们的含义。

假设一共转动幸运轮$k$次，而其中有$k_i$次转动的结果为$m_i$。你所得到的总钱数为：$\sum\limits_{i=1}^{n}m_i k_i$，那么每次转动的钱数为$M=\frac{\sum\limits_{i=1}^{n}{m_i k_i}}{k}$，现在假设$k$是一个很大的数字，那么我们可以假设概率与频率相互接近。即：

$$\frac{k_i}{k}\approx p_i, i=1,\ldots,n$$

这样你每次转动幸运轮所期望得到的钱数是：

$$M=\frac{\sum\limits_{i=1}^{n}m_i k_i}{k}\approx \sum\limits_{i=1}^{n}m_i p_i$$

有这个例子启发，才有了下面的定义。

## 期望的定义

设随机变量$X$的概率函数是$p_X$，那么$X$的期望值（也称期望或均值）为：

$$E[X]=\sum\limits_{x}xp_X(x)$$

虽然内容较为简单，但是用频率接近概率进而引进概率的定义是很常见的思路，有了这个过程我们对期望才有了很直观的理解。
---
title: 经典摘录-几何随机变量的均值和方差
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-23 22:16:00
tags: [probability]
toc: true
copyright: false
---

本文摘录自 [Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html)  Example 2.17 mean_and_variance_of_the_geometric

你一次又一次地写一个计算机软件，每写一次都有一个成功的概率 $p$ 。假设每次成功与否与以前的历史记录相互独立。令 $X$ 是你一直到成功为止所写的次数（最后一次你成功了！） $X$ 的期望和方差是多少？

由于 $X$ 是一个几何随机变量，那么我们视 $X$ 为几何随机变量，概率质量函数是：

$$p_X(k)=(1-p)^{k-1}p, k = 1, 2, ....$$

那么 $X$ 的方差和均值为：

$$E[X] = \sum\limits_{k=1}^{\infty}k(1-p)^{k-1}p, var(X)=\sum\limits_{k=1}^{\infty}(k-E[X])^2(1-p)^{k-1}p$$

但是衡量这些无限和有点麻烦。我们利用全期望定理进行计算。记 $A_1=\{X=1\}=\{\text{first try is a success}\}, A_2=\{X>1\}=\{\text{first try is a failure}\}$。 如果第一次就成功，得到 $X=1​$ ，且

$$E[X|X=1]=\sum\limits_{}^{}xp_{X|X=1}=1p_{1|X=1}=1$$ 

如果首次尝试失败 ( X > 1)，我们将浪费一次尝试，我们重新开始，由于是在第一次失败的条件下，那么表示尝试次数的 $X$ 的均值一定是大于1的，剩余尝试的期望即 $E[X]$  。

$$E[X|X>1] = E[X+1] = 1+E[X]$$

因此，由全期望定理：

$$
\begin{eqnarray}
E[X] &=& P[X=1]E[X|X=1]+P(X>1)E[X|X>1] \\
&=& p + (1 - p) (1+E[X])  
\end{eqnarray}
$$
从而可以得到：

$$E[X]=\frac{1}{p}$$

相似的推理，我们也得到

$$E[X^2|X=1]=1,\quad E[X^2|X>1]=E[(1+X)^2]=1+2E[X]+E[X^2]$$

因此，

$$E[X^2]=p+(1-p)(1+2E[X]+E[X^2])$$

联合 $E[X]=\frac{1}{p}$ 得到：

$$E[X^2]=\frac{2}{p^2}-\frac{1}{p}$$

总结：

$$var(X)=E[X^2]-(E[X])^2= \frac{2}{p^2}-\frac{1}{p}-\frac{1}{p^2}=\frac{1-p}{p^2}$$


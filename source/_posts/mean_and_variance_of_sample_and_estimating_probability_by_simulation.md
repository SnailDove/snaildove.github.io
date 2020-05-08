---
title:  经典摘录-样本均值的期望方差与模拟的方法估计概率
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-25 20:16:00
commets: true
toc: true
copyright: false
---

## 样本均值的期望方差

摘录自：[Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html) Example 2.21. Mean and Variance of the Sample Mean  

我们希望估计总统的支持率。为此，我们随机地选取n个选民，询问他们的看法。令 $x_i​$ 表示 $i​$ 个被问的选民的态度：

$$X_i = \cases{1, \text{若第 $i$ 个被问的选民支持总统}\\0, \text{若第 $i$ 个被问的选民不支持总统}}$$ 

**假设$X_1,\ldots, X_n$为独立同分布的伯努利随机变量**，其均值为 $p$，方差为 $p(1-p)$ 。此处我们将 $p$ 认为选民支持总统的概率，并且将对调查得到的回应进行平均处理，计算样本均值 $S_n$ ，把 $S_n$ 定义为 

$$S_n=\frac{X_1+ \ldots + X_n}{n}$$

因此，随机变量 $S_n$ 是对n个选民抽样的支持率。

由于 **$S_n$ 是 $X_1, \ldots, X_n$ 的一个线性函数**，我们**利用均值的线性关系**得到，

$E[S_n]=\sum\limits_{i=1}^{n} E[\frac{X_i}{n}]=\sum\limits_{i=1}^{n}\frac{1}{n}E[X_i]= \sum\limits_{i=1}^{n}\frac{1}{n}p=p=E[X]$

再**利用$X_1,\ldots, X_n$ 的独立性**，可以得到：

$$var(S_n) = \sum\limits_{i=1}^{n}var(\frac{X_i}{n}) = \sum\limits_{i=1}^{n}\frac{1}{n^2}var(X_i) = \frac{p(1-p)}{n}$$

**样本均值为 $S_n$ 被认为是支持率很好的估计，这是因为它的期望值刚好是 $p$。然后反映精度的方差随着样本大小的$n$ 增大的时候，变得越来越小。** 

注意，上例中即使 $X_i$ 不是伯努利随机变量，结论

$$var(S_n) = \frac{var(X)}{n}$$

仍然成立，只要 $X_i$ 之间相互独立，毕竟期望和方差与 $i$ 无关。因此，随着样本大小增加，样本均值仍然是随机变量的均值的一个很好的估计。我们将在第5章再详细讨论样本均值的这些属性，并且与大数定律结合起来。

## 模拟的方法估计概率 

摘录自：[Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html) Example 2.22. Estimating Probabilities by Simulation 

在许多实际问题中，有时候计算一个事件的概率是十分困难的，然后我们可以用物理方法或计算机方法重复地进行试验，这些试验结果可以显示事件是否发生。利用这种模拟方法可以以很高的精度计算某事件的概率。可以独立地模拟试验 $n$ 次，并且记录 $n$ 次试验中的 $A$ 发生的次数 $m$，用 $\frac{m}{n}$ 去近似概率 $P(A)$。例如在抛掷硬币试验中，计算概率 $p=P$ （出现正面），独立地抛掷 $n$ 次，用比值（记录中出现的正面次数除以试验总次数n）去逼近概率$p$。

为计算这种方法的精确度，考虑 $n$ 个独立同分布的伯努利随机变量 $X_1,\ldots, X_n$，每个 $X_i$ 的概率质量函数：

$$p_{X_i}(k)=\cases{P(A), if\ k=1\\1-P(A), if\ k=0}$$

在模拟环境中，$X_i$ 有关于第 $i$ 次试验的结果，如果第 $i$ 次的试验结果属于 $A$ ，那么 $X_i$ 取值为1，那么随机变量的取值（$$X=\frac{X_1+X_2+\ldots+X_n}{n}$$） 就是概率 $P(A)$ 的估计值。由例 2.21 的结果知，$X$ 的期望为 $P(A)$，方差为 $\frac{P(A)(1-P(A))}{n}$。故 $n$ 很大时，$X$提供了 $P(A)$ 的精确的估计。

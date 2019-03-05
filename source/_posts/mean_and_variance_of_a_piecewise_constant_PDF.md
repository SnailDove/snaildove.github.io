---
title:  经典摘录-分段常数概率密度函数的均值和方差 
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-27 20:16:00
commets: true
toc: true
---

本文摘录自 [Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html) Example 3.17 Mean and Variance of a Piecewise Constant PDF

假设一个随机变量 $X$ 有分段常数的概率密度函数

$$f_x(x)=\cases{\frac{1}{3}, & if $0 \le x \le 1$, \\ \frac{2}{3}, & if $ 1 < x \le 2$, \\0, & if otherwise}$$

![Figure3.14_Piecewise_constant_PDF_for_Example3.17.png](http://pne0wr4lu.bkt.clouddn.com/gitpage/introduction-to-probability/mean_and_variance_of_a_piecewise-constant_PDF/2.png)

考虑事件：

$$A_1=\{\text{X 位于第一个区间 [0,1]}\}$$

$$A_2=\{\text{X 位于第二个区间 (1,2]}\}$$

我们从已知的概率密度函数得到：

$$P(A_1)=\int_{0}^{1}f_X(x)dx=\frac{1}{3}, \quad P(A_2)=\int_{1}^{2}f_X(x)dx=\frac{2}{3}$$

因此，条件均值和 $X$ 的条件二阶矩容易计算，因为相关的概率密度函数 $PDF_S$：  $f_{X|A_1}$ 和 $f_{X|A_2}$ 是均匀的，回忆例子3.4得到， 均匀分布在区间 $[a,b]$ 上的的随机变量的均值是：$\frac{(a+b)}{2}$ ，它的二阶矩是 $\frac{(a^2+ab+b^2)}{3}$ ，因此：

$$
\begin{eqnarray}
E[X|A_1]&=&\frac{1}{2},\quad E[X|A_2]&=&\frac{3}{2}\\
E[X^2|A_1]&=&\frac{1}{3},\quad E[X^2|A_2]&=&\frac{7}{3}
\end{eqnarray}
$$

使用总期望定理得到：
$$
\begin{eqnarray}
E[X] &=& P(A_1)E[X|A_1]+P(A_2)E[X|A_2] &=& \frac{1}{3} \cdot \frac{1}{2}+\frac{2}{3}\cdot\frac{3}{2} &=& \frac{7}{6} \\
E[X^2] &=& P(A_1)E[X^2|A_1]+P(A_2)E[X^2|A_2] &=& \frac{1}{3}\cdot\frac{1}{3}+\frac{2}{3}\cdot\frac{7}{3} &=& \frac{15}{9}
\end{eqnarray}
$$
那么可以得到方差：

$$var(x)=E[X^2]-(E[X])^2=\frac{15}{9}-\frac{49}{36}=\frac{11}{36}$$

**注意：** 对于计算均值和方差的方法是容易推广到多分段的常数概率密度函数。

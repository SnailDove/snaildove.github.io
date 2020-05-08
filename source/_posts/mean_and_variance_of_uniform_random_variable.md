---
title:  经典摘录-均匀随机变量的均值和方差 
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-26 20:16:00
commets: true
toc: true
---

说明：全文摘自[Introduction to robability, 2nd Edition](http://www.athenasc.com/probbook.html) 

## 均匀分布的离散随机变量

按照定义，离散均匀随机变量的取值范围由相邻的整数所组成的有限集合，而取每个整数的概率都是相等的。这样它的分布列：

$$p_X(k)=\cases{\frac{1}{b-a+1}, & if k=a, a+1, ... ,b\\0, &  otherwise}$$

![mean_and_variance_of_the_discrete_uniform_random_variable.png](http://q9kvrafcq.bkt.clouddn.com/gitpage/introduction-to-probability/mean_and_variance_of_uniform_random_variable/1.png)

其中$a,b$ 是两个整数，作为随机变量的值域的两个端点，$a<b$。由于它的概率函数相对于(a+b)/2 是对称的，所以其均值为：

$$E[X]=\frac{a+b}{2}$$

为计算$X$的方差，先考虑a=1和b=n的简单情况。利用归纳法可以证明：

$$E[X^2]=\frac{1}{n}\sum\limits_{k=1}^{n}k^2=\frac{1}{6}(n+1)(2n+1)$$

（具体证明过程留作习题）。这样利用一、二阶矩，可得到$X$的方差
$$
\begin{eqnarray\*}
var(X)&=& E[X^2]-(E[X])^2\\
&=&\frac{1}{6}(n+1)(2n+1)-\frac{1}{4}(n+1)^2\\
&=&\frac{n^2-1}{12}
\end{eqnarray\*}
$$
**对于 $a$ 和 $b$ 的一般情况，实际上在区间 $[a,b]$上的均匀分布与在区间 $[1,b-a+1]$ 上的分布之间的差异，只是一个分布是另外一个分布的偏移，因此两者具有相同的方差（此处区间 $[a,b]$ 是指处于 $a$ 和 $b$ 之间的整数的集合）**。这样在一般的情况下，$X$ 的方差只需将简单的情况下公式中的 $n$ 替换成 $b-a+1$ ，即：

$$var(X)=\frac{(b-a+1)^2-1}{12}=\frac{(b-a)(b-a+2)}{12}$$

## 均匀分布的连续随机变量

摘录自 Example 3.4. Mean and Variance of the Uniform Random Variable 

设随机变量 $X$ 的分布为 $[a,b]$ 上的均匀分布，得到：
$$
\begin{eqnarray\*}
E[X] &=& \int_{-\infty}^{+\infty}xf_X(x)dx \\
&=& \int_{a}^{b}x\frac{1}{b-a}dx \\
&=& \frac{1}{b-a}\cdot \frac{1}{2}x^2|^{b}_{a} \\
&=& \frac{1}{b-a}\cdot\frac{b^2-a^2}{2} \\
&=& \frac{b+a}{2}
\end{eqnarray\*}
$$
这个期望值刚好等于 $PDF$ 的对称中心 $\frac{b+a}{2}$ 。

为求得方差，先计算 $X$ 的二阶矩：
$$
\begin{eqnarray\*}
E[X^2] &=& \int_{a}^{b}\frac{x^2}{b-a}dx \\
&=& \frac{1}{b-a}\cdot\int_{a}{b}x^2dx \\
&=& \frac{1}{b-a}\cdot \frac{1}{3}x^3|_{a}^{b} \\
&=& \frac{b^3-a^3}{3(b-a)} \\
&=& \frac{a^2+ab+b^2}{3} \\
\end{eqnarray\*}
$$
这样随机变量 $X$ 的方差为：
$$
var(X)=E[X^2]-(E[X])^2=\frac{a^2+ab+b^2}{3}-\frac{(a+b)^2}{4}=\frac{(b-a)^2}{12}
$$

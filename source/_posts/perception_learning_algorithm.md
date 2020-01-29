---
title: Perception Learning Algorithm
mathjax: true
mathjax2: true
categories: 中文
date: 2015-04-24 20:16:00
tags: [Machine Learning]
commets: true
toc: true
top: true
---

PLA(Perception Learning Algorithm) 适用于二维及高维的线性可划分问题。问题的答案只有同意或者不同意。

## 例子

银行可以根据顾客的个人信息来判断是否给顾客发放信用卡。**将顾客抽象为一个向量$X$，包括姓名、年龄、年收入、负债数等。同时设定各个属性所占的权重向量为$W$，对于正相关的属性设置相对较高的权重，如年收入，对于负相关的属性设置较低的权重，如负债数。$y$表示是否想该用户发放了信用卡。**通过求$X$和$W$的内积减去一个阀值threshold，若为正则同意发放信用卡，否则不发放信用卡。我们假设存在着一个从$X$到$Y$的映射$f$，PLA算法就是用来模拟这个映射，使得求出的函数与$f$尽可能的相似，起码在已知的数据集(即样本上)上一致。

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/1.png)

**PLA算法即用来求向量$W$**，使得在已知的数据中机器做出的判断与现实数据相同。当$X$为二维向量时，相当于在平面上画出一条直线将所有的点分成两部分，一部分同意发送，另一部分的不同意。内积可以表示成：
$$
\begin{eqnarray}
h(x) &=& sign((\sum\limits_{i=1}^{d}W_i X_i)-threshold)\\
&=& sign((\sum\limits_{i=1}^{d}W_i X_i)+\underbrace{(-threshold)}_{W_0}\times\underbrace{(+1)}_{X_0})\\
&=& sign(\sum\limits_{i=0}^{d}W_i X_i)\\
&=& sign(W^TX)
\end{eqnarray}
$$

其中$X_0=1，W_0=-threshold$

$y_s$的值域：$\{+1，-1\}$，($y_s$ 表示样本中$y$的值，用于输入到算法进行调整)

结合文中例子：$y_s=1$ 表示在给定的样本数据中，给该用户发放了信用卡，$y_s= -1$表示未发放。

PLA先假定$W_0$为向量$\vec{0}$，然后找到一个不满足条件的点，调整$W$的值，依次进行**迭代所有样本数据**使得最终可以将两部分完全分开。

## W的调整方案

**错误驱动调整**

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/2.png)

解释一下ppt的内容，出现错误分2种情况：

1.  在给定的已知数据中向该用户发放了数据，即$y_s(i)$样本中第$i$个数据为$+1$，但算法给出的结果是不发放（$h(X_i) <0$），说明两个向量的内积为负，需要调整$W$向量使得两条向量更接近，此时令调整系数为样本的$y_s(i)$，则调整后的$W_{t+1}= W_t + y_s(i)X_i$，$W$的下标$t, t+1$表示调整的次数，示意图:

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/3.png)


2.  在给定的已知数据中向该用户发放了数据，即$y_s(i)$样本中第$i$个数据为$-1$，但算法给出的结果是不发放（$h(X_i) > 0$），说明两个向量的内积为正，需要调整$W$向量使得两条向量更远离，此时令调整系数为样本的$y_s(i)$，则调整后的$W_{t+1}= W_t + y_s(i)X_i$，示意图:

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/4.png)

**注意：2种不同情况的调整的表达式都一样**

## 对于线性可分的数据集，PLA算法是可收敛的

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/5.png)

**两个向量的内积增大说明：**

1.  **两个向量夹角越小**
2.  **或者向量的长度增大**

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/6.png)

老师的ppt上 $||W_{t+1}||^2  \le  ||W_t||^2 + max\{1 \le i \le  n\ \ |\ \ ||y_i X_i||^2\}$ 其中，$y_i$的值域 $\{+1, -1\}$

因此 $||W_{t+1}||^2  \le  ||W_t||^2 + max\{1 \le i \le n\ \ |\ \ ||X_i||^2\}$

这说明每次调整后，向量的长度增加有限。不妨

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/7.png)

带入上一公式得到：

$$
\begin{eqnarray}
\frac{||W_{t+1}||^2}{||W_{t}||^2} &\le& \frac{||W_{t}||^2 + max||y_n x_n||^2}{||W_{t}||^2} \\
&=& \frac{||W_{t}||^2 + max||x_n||^2}{||W_{t}||^2} \\
&=&1+ \frac{R^2}{||W(t)||^2}
\end{eqnarray}
$$

因此$W_t$最终是收敛的，到此已经证明了PLA算法最终可以停止。

## 算法需要调整的次数

由上述过程可以得到以下两个不等式：

$$
\begin{eqnarray}
W_f^t W_t &=& W_f^t(W_{t-1}+y_s(t-1)X_{t-1}) \\
&=& W_f^tW_{t-1}+y_s(t-1)W_f^tX(t-1) \\
&\ge& W_f^tW_{t-1}+min(yW_f^tX) \\
&\ge& W_f^tW_{t-2}+2\,min(yW_f^tX) \\
&\cdots& \\
&\ge& W_f^tW_0 + t\,min(yW_f^tX)  \\
&=& t\,min(yW_f^tX)
\end{eqnarray}
$$

$$
\begin{eqnarray}
||W_t||^2 &\le& ||W_{t-1}||^2+max(||X||^2)\\
&\le& ||W_{t-2}||^2+2\,max(||X||^2)\\
&\cdots& \\
&\le& ||W_0||^2+t\,max(||X||^2)\\
&=& t\, max(||X||^2)
\end{eqnarray}
$$

那么来看这个式子：$\frac{W_f^t  W_t}{||W_f^t||\ ||W_t||}\ge \frac{t\ min(yW_f^tX)}{||W_f^t||\sqrt{t\,(max(||X||))^2}}=\sqrt{t}\, \frac{min(yW_f^tX)}{||W_f^t||max(||X||)} $  

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/8.png)

再根据余弦值最大为1，可以得到$\frac{W_f^tW_t}{||W_f^t||\ ||W_t||}\le 1$，于是我们得到调整次数：$t\le \frac{||W_f^t||(max(||X||))^2}{(min(yW_f^tX))^2}={R^2 \over \rho^2}$.

## PLA的优缺点

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/9.png)

一方面，我事先肯定不知道$W_f^t$，另一方面为了应对可能出现的噪声。那么怎么衡量当前得到的直线能够满足要求呢？我们只能在每一步的时候都判断一下，调整后的$W_{t+1}$**是否**比上一次的$W_t$能够线性可分更多的数据，于是有了下面的改进算法Pocket PLA，PocketPLA比PLA在调整的时候多做一步：判断当前改正犯的错是否比之前更小，也就是贪心选择。

## Pocket PLA

![img](http://q4vftizgw.bkt.clouddn.com/gitpage/Hsuan-Tien_Lin/perception-learning-algorithm/10.png)

参考

1.  [HappyAngel](http://www.cnblogs.com/HappyAngel/p/3456762.html)
2.  [DreamerMonkey](http://blog.csdn.net/dreamermonkey/article/details/44065255)
3.  ppt全部来自台大《机器学习基石》课堂

---
title:  经典摘录-[累积]分布函数CDF 
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-26 20:16:00
commets: true
toc: true
---

说明：全文摘自 [Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html)

## 分布函数

我们分别用概率质量函数 PMF(Probability Mass Function) 和概率密度函数 PDF(Probability Density Function) 来刻画随机变量 $X$ 的取值规律。现在**希望用一个统一的数学工具去刻画随机变量的取值规律**。  

​**分布函数**（用记号 CDF 表示简称）就能完成这个任务。 $X$ 的 CDF 是一个 $x$ 的函数，对每一个 $x$ ，$F_X(x)$ 定义为 $P(X\le x)$ 。特别地，当 $X$ 为离散或连续的情况下：
$$
F_X(x)=P(X\le x)=\cases{\sum\limits_{k\le x}p_X(k), \text{若 $X$ 是离散的}\\\int_{-\infty}^{x}f_X(x)dt, \text{若 $X$ 是连续的 }}
$$
**分布函数又称为累积分布函数（cumulative distribution function）**，累积意味着 $F_X(x)$ 将 $X$ 取值的概率由 $-\infty\rightarrow x$。 

在一个概率模型中，随机变量可以有不同的类型，可以是离散的，也可以是连续的，甚至可以是既非离散的也非连续的。但不管是什么类型的随机变量，它们有一个共同的特征，即都有一个分布函数，这是因为 $\{X\le x\}$ 是一个随机事件，这些事件的概率形成概率分布。今后，凡是通过 PMF\PDF\CDF刻画事件 $\{X\le x\}$ 概率的，都称为**随机变量 $X$ 的概率律**。因此离散情况下的分布列，连续情况下的概率密度函数以及一般情况下的分布函数都是相应的随机变量的概率律。

下图分别给出不同的离散随机变量和连续随机变量的 CDF 的一些说明。从这些图像以及 CDF 的定义，可以得到 CDF 的某些性质。

![Figure_3.6_CDFs_of_some_discrete_variables.png](http://q9kvrafcq.bkt.clouddn.com/gitpage/introduction-to-probability/cumulative-distribution-function/1.png)

上图这些离散随机变量的 CDF ，通过随机变量的概率质量函数（PMF）可求得相应的分布函数：
$$
F_X(x)=P(X\le x)=\sum\limits_{k\le x}p_{X}(k)
$$
这个函数是一个阶梯函数，在具有正概率的那些点上具有跳跃。在跳跃点上， $F_X(x)$ 取较大的那个值，即 $F_X(x)$ 保持右连续。

![Figure_3.7_CDFs_of_some_continuous_random_variables.png](http://q9kvrafcq.bkt.clouddn.com/gitpage/introduction-to-probability/cumulative-distribution-function/2.png)

上图的这些连续随机变量的 $CDF$ 。通过随机变量的概率密度函数（PDF）可求得相应的分布函数：
$$
F_X(x)=P(X\le x)=\int_{-\infty}^{+\infty}f_X(t)dt
$$
概率密度函数 $f_X(x)$ 可由 CDF 经求微分得到：
$$
f_X(x)=\frac{dF_X(x)}{dx}(x)
$$
对于连续随机变量，CDF 是连续的

## CDF 的性质

假设 $X$ 的 CDF $F_X(x)$ 是由下式定义的 ：
$$
F_X(x)=P(X\le x), \forall x
$$
并且 $F_X(x)$ 具有下列性质：

1.  $F_X(x)$ 是 $x$ 的单调非减函数：若 $x\le y$ ，则 $F_X(x)\le F_X(y)$ 。

2.  当 $x\rightarrow -\infty$ 的时候，则 $F_X(x)\rightarrow 0$ ，当 $x\rightarrow +\infty$ ，则 $F_X(x)\rightarrow 1$ 。

3.  当 $X$ 是离散随机变量的时候， $F_X(x)$ 为阶梯函数。

4.  当 $X$ 是连续随机变量的时候， $F_X(x)$ 为 $x$ 的连续函数。

5.  当 $X$ 是离散随机变量并且取整数数值的时候，分布函数和概率质量函数（PMF）可以利用求和或差分互求：
    $$
    F_X(k)=\sum\limits_{i=-\infty}^{k}p_X(i)\\
    p_X(k)=P(X\le k)-P(X\le k-1)=F_X(k)-F_X(k-1)
    $$
    其中 $k$ 可以是任意整数。 

6.  当 $X$ 是连续随机变量的时候，分布函数与概率密度函数可以利用积分和微分互求：
    $$
    F_X(x)=\int_{-\infty}^{x}f_X(t)dt,\quad f_X(x)=\frac{dF_X}{dx}(x)
    $$
    (第二个等式只在分布函数可微的那些点上成立)

**有时候为了计算随机变量的概率质量函数或概率密度函数，首先计算随机变量的分布函数会更方便些**。在连续随机变量的情况下，将在4.1节系统地介绍用该方法求随机变量的函数的分布。下面是一个离散随机变量的计算例子。

## 例子

### 几个随机变量的最大值

你参加某种测试，按规定三次测试的最高成绩作为你的最后成绩，设 $X=max\{X_1,X_2,X_3\}$ ，其中 $X_1,X_2,X_3$ 是三次测试成绩，$X$ 是你的最后的成绩。假设你的每次测试成绩是 1 分到 10 分之间，并且 $P(X=i)=\frac{1}{10}, i=1,...,10$ 。现在求最终成绩 $X$ 的概率质量函数。

采用间接方法求分布函数。首先计算 $X$ 的 CDF，然后通过
$$
p_X(k)=F_X(k)-F_X(k-1), i=1,\ldots,10
$$
得到 $X$ 的概率质量函数。对于 $F_X(k)$ ，得到：
$$
\begin{eqnarray}
F_X(k) &=& P(X\le k) \\
&=& P(X_1\le k, X_2\le k, X_3\le k) \\
&=& P(X_1\le k)P(X_2\le k)P(X_3\le k) \\
&=& (\frac{k}{10})^3
\end{eqnarray}
$$
此处第三个等式是由事件 $\{X_1\le k\},\{X_2\le k\},\{X_3\le k\}$ 相互独立所致。这样 $X$ 的概率质量函数为：
$$
p_X(k)=(\frac{k}{10})^3-(\frac{k-1}{10})^3, k=1,\ldots,10
$$
本例的方法可推广到 $n$ 个随机变量 $X_1,\ldots,X_n$ 的情况。如果对每一个 $x$ ，事件 $\{X_1\le x\},\ldots, \{X_n\le x\}$ 相互独立，则 $X=max\{X_1,\ldots,X_n\}$ 的 CDF 为：
$$
F(x)=F_{X_1}(x)\cdots F_{X_n}(x)
$$
利用这个公式，在离散情况下通过差分可得到 $P_X(x)$ ，在连续情况下通过微分可得到 $f_X(x)$ 。

### 距离的分布函数和概率密度函数

习题3.5 ：按照均匀分布律，在一个三角形中随机的选取一个点，设已知三角形的高，求这个点到底边的距离 $X$ 的分布函数和概率密度函数。

用 $b$ 表示底的长度，$h$ 表示三角形的高度，$A=\frac{bh}{2}$ 表示三角形的面积。随机地在三角形内选取一个点，然后画一条平行于三角形底边的辅助直线，用 $A_x$ 表示由这条辅助线构成的小三角形的面积，那么这个小三角形的高度即 $h-x$ ，它的底边按比例求得：$b\frac{h-x}{h}$ ，因此 $A_x=\frac{b(h-x)^2}{2h}$ 。对于 $x\in [0,h]$ ，得到：
$$
F_X(x)=P(0< x \le x)=1-P(X>x)=1-\frac{A_x}{A}=1-\frac{\frac{b(h-x)^2}{2h}}{\frac{bh}{2}}=1-(\frac{h-x}{h})^2
$$
当 $x<0,$ 那么 $F_X(x)=0$ ; 当 $x>h,$ 那么 $F_X(x)=1$ 。

概率密度函数可以对累积分布函数 CDF 进行求微分得到：
$$
f_X(x)=\frac{dF_X}{dx}(x)=\cases{\frac{2(h-x)}{h^2}, & 当 $0\le x \le h$\\0, & 其他情况}
$$

### 等待时间

习题3.6 ：Jane去银行取款，有1个或0个顾客在她前面，这两种情况是等可能的。已知一个顾客的服务时间是一个指数随机变量，参数为 $\lambda$ 。那么Jane所等待的时间分布函数是？

用 $X$ 表示等待的时间，用 $Y$ 表示在Jane之前顾客的数量。于是得到：$\forall x <0, F_X(x)=0$ ，其他情况下，根据题意得：
$$
F_X(x)=P(X\le x)=\frac{1}{2}P(X\le x| Y=0)+\frac{1}{2}P(X\le x|Y=1)
$$
又因为
$$
P(X\le x|Y=0)=1,\quad P(X\le x|Y=1)=1-e^{-\lambda x}
$$
得到
$$
F_X(x)=\cases{\frac{1}{2}(2-e^{-\lambda x}), & if $x \ge 0$ \\0, & 其他情况}
$$
注意：这个累积分布函数 CDF 在 $x=0$ 处连续，随机变量 $X$ 既不是离散的也不是连续的。

### 投飞标游戏

Alvin在进行飞镖游戏，飞镖的靶是一块半径为 r 的圆板。记 $X$ 为飞镖的落点到靶心的距离。假设落点在靶板上均匀地分布。

(a) 求出 $X$ 的概率密度函数、均值和方差。

$X$ 的累积分布函数比较容易求得：
$$
F_X(x)=\cases{
P(X\le x)=\frac{\pi x^2}{\pi r^2}=(\frac{x}{r})^2, & if $\forall x\in [0,r]$\\
0, & if $x < 0$\\
1, & if $x>r$
}
$$
通过微分，得到概率密度函数：
$$
f_X(x)=\cases{
\frac{2x}{r^2}, & if $0\le x\le r$\\
0, & otherwise
}
$$
进而通过积分得到：
$$
E[X]=\int_{0}^{r}\frac{2x^2}{r^2}dx=\frac{2r}{3}\\
E[X^2]=\int_{0}{r}\frac{2x^3}{r^2}dx=\frac{r^2}{2}\\
var(X)=E[X^2]-(E[X])^2=\frac{r^2}{2}-\frac{4r^2}{9}=\frac{r^2}{18}
$$
(b) 靶上画出了一个半径为 $t$ 的同心圆。若 $X\le t$ ，Alvin的得分为 $S=\frac{1}{X}$ ，其他情况 $S=0$ 。求出 $S$ 的分布函数。 $S$ 是不是连续随机变量？

由题意得：当且仅当 $X\le t$ ，Alvin 获得一个介于 $[\frac{1}{t}, +\infty)$ 的分数s，其它情况下，他的得分为 0 。因此：
$$
F_S(s)=\cases{ 
0, \quad \text{if $s<0$}\\
P(S\le s)=1-P(X\le t), \quad \text{if $0\le s\le \frac{1}{t}$  (即Alvin击中了内圆之外)} \\
P(S\le s)=P(X\le t)P(S\le s|X\le t)+P(X>t)P(S\le s|X>t) \quad \text{if $s > \frac{1}{t}$}
}
$$
根据题意，得到：
$$
P(X\le t)=\frac{t^2}{r^2},\quad P(X>t)=1-\frac{t^2}{r^2}
$$
而且因为当 $X>t, S=0$ ， 所以 $P(S\le s|X> t)=1$ 。

进而得到：
$$
P(S\le s| X\le t)=P(\frac{1}{X}\le s|X\le t)=P(\frac{1}{s}\le X|X\le t) = \frac{P(\frac{1}{s}\le X \le t)}{P(X\le t)} =\frac{\frac{\pi t^2 -\pi(\frac{1}{s})^2}{\pi r^2}}{\frac{\pi t^2}{\pi r^2}}=1-\frac{1}{s^2t^2}
$$
最后得到：
$$
F_S(s)=\cases{
    0, & \text{if }s<0 \\
    1-\frac{t^2}{r^2}, & \text{if } 0\le s \le \frac{1}{t}\\
    1-\frac{1}{s^2r^2} & \text{if } \frac{1}{t}<s
}
$$
因为 $F_S(s)$ 在 $s=0$ 处不连续，所以随机变量 $S$ 不是连续的。

## 几何和指数随机变量的分布函数

**由于分布函数对一切随机变量都适用，可以利用它来探索离散和连续随机变量之间的关系**。特别地，此处讨论几何随机变量和指数随机变量之间的关系。

设 $X$ 是一个几何随机变量，其参数为 $p$ ，即 $X$ 是在伯努利独立试验序列中直到第一次成功所需要的试验次数，而伯努利试验的参数为 $p$ 。这样对于 $k=1,2\cdots,$  得到 $P(X=k)=p(1-p)^{k-1}$ ，而 $X$ 的 CDF 为：
$$
F_{geo}(n)=\sum\limits_{k=1}^{n}p(1-p)^{k-1}=p\frac{1-(1-p)^n}{1-(1-p)}=1-(1-p)^n,\quad n=1,2,\cdots
$$
现在设 $X$ 是一个指数随机变量，其参数 $\lambda>0$ 。其 CDF 是
$$
F_{exp}(x)=P(X\le x)=0,\quad x\le 0\\
F_{exp}(x)=\int_{0}^{x}\lambda e^{-\lambda t}dt=-e^{-\lambda t}|^{x}_{0}=1-e^{-\lambda x},\quad x>0
$$
现在比较两个分布函数，令 $\delta=\frac{-ln(1-p)}{\lambda}\rightarrow \delta\lambda=-ln(1 - p)$ ，这样得到：
$$
e^{-\lambda\delta}=1-p \quad (\*)
$$
那么，将 $(*)​$ 代入 $F_{geo}(n)​$ 得：$1-(e^{-\lambda\delta})^n=1-e^{-n\lambda\delta}​$ ，而分布函数 $F_{exp}​$ 在 $x=n\delta​$ 处为： $1-e^{-\lambda n\delta}=1-e^{-n\lambda\delta}​$ 是与 $F_{geo}​$ 在 $n​$ 处相等的，$n=1,2,\cdots​$ ，即：
$$
F_{exp}(n\delta)=F_{geo}(n)=, n=1,2,\cdots
$$
现在假定以很快的速度抛掷一枚不均匀的硬币 (每 $\delta$ 秒抛掷一次，$\delta \ll 1$)，每次抛掷，正面向上的概率为 $p=1-e^{-\lambda\delta}$ 。这样第一次得到正面向上所抛掷的次数为 $X$ ，第一次得到正面向上的时刻为 $X\delta$ ，$X\delta$ 与参数为 $\lambda$ 的指数随机变量十分接近，这只需看它们的分布函数即可（看下图）。这在本书第六章讨论伯努利和泊松分布过程的时候，这种关系显得特别重要。
![Figure_3.8_Relation_of_the_geometric_and_the_exponential_CDFs.png](http://q9kvrafcq.bkt.clouddn.com/gitpage/introduction-to-probability/cumulative-distribution-function/3.png)

几何随机变量和指数随机变量的分布函数之间的关系。图中离散分布函数为 $X\delta$ 的分布函数，$X$ 是参数为 $p=1-e^{-\lambda x}$ 的几何随机变量。当 $\delta\rightarrow 0$ 时，$X\delta$ 的分布函数趋于指数分布函数 $1-e^{-\lambda x}$ 。

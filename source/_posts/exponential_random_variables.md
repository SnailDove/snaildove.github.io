---
title:  经典摘录-指数型随机变量的均值和方差 
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-25 20:16:00
commets: true
toc: true
---
**说明**：全文摘自 [Introduction to probability, 2nd Edition](http://www.athenasc.com/probbook.html) 

一个指数型随机变量是拥有以下形式的**概率密度函数**：
$$
f_X(x)=\cases{
    \lambda e^{-\lambda x}, & if $x\ge 0$ \\
    0, & 其他情况
}
$$
这个公式中**$\lambda$ 是一个正数**，这是一个符合概率归一性的定义，因为：
$$
\int_{-\infty}^{+\infty}f_X(x)dx=\int_{-\infty}^{+\infty}\lambda e^{-\lambda}dx=-e^{\lambda x}|_0^{+\infty}=1
$$
注意，指数型概率密度函数具有这样的特性：$X$ 超过某个值的概率随着这个值的增加而按指数递减
$$
\forall  a\ge 0,P(X\ge a)=\int_{a}^{\infty}\lambda e^{-\lambda x}dx=-e^{-\lambda x}|_a^{+\infty}=e^{-\lambda a}
$$
由概率密度函数得到**累积分布函数**：
$$
\forall x \ge 0, P(X\le x)=\int_{0}^{x}\lambda e^{-\lambda x}dx=1-e^{-\lambda x}
$$
![The_PDF_of_an_exponential_random_variable](http://q4vftizgw.bkt.clouddn.com/gitpage/introduction-to-probability/exponential-random-variable/1.png)

**指数型随机变量能够对直到事件发生的时间建模，例如：消息到达计算机的时间，设备的使用寿命，灯泡的寿命，事故发生的时间等等**（An exponential random variable can, for example, be a good model for the amount of time until an incident of interest takes place）。将在后面的章节看到指数型随机变量与几何随机变量紧密关联，几何随机变量也与相关事件发生的（离散）时间相关联。**指数型随机变量将在第六章随机过程的学习中扮演重要的角色**。但是目前为止，仅仅视它为一种特殊的可分析追中的随机变量。

**指数型随机变量的均值和方差**：
$$
\begin{eqnarray}
E[X] &=& \int_{0}^{\infty}x\lambda e^{-\lambda x}dx \\
&=& (-xe^{-\lambda x})|_0^{\infty} + \int_{0}^{\infty}e^{-\lambda x}dx \quad\text{这一步利用分部积分法}\\
&=& 0-\frac{e^{-\lambda x}}{\lambda}|_0^{\infty}\\
&=& \frac{1}{\lambda}
\end{eqnarray}
$$
在此利用分布积分法，可得到 $X$ 的二阶矩：
$$
\begin{eqnarray}
E[X^2] &=& \int_{0}^{\infty}x^2\lambda e^{-\lambda x}dx \\
&=& (-x^2e^{-\lambda x})|_0^{\infty}+\int_{0}^{\infty}2xe^{-\lambda x}dx
&=& 0+ \frac{2}{\lambda}E[X]\\
&=& \frac{2}{\lambda^2}
\end{eqnarray}
$$
最后利用公式 $var(x)=E[X^2]-(E[X])^2$ ，得到：
$$
var(X)=\frac{2}{\lambda^2}-\frac{1}{\lambda}=\frac{1}{\lambda^2}
$$

### 例3.5 

小陨石落入非洲撒哈拉沙漠的时间是遵从指数族分布的。具体地说，从某一观察者开始观察，知道发现一个陨石落到沙漠中，这个时间被模拟成指数型随机变量，其均值为 $10$ 天，现在假定，目前时间为晚上 $12$ 点整。问第二天早晨 $6:00$ 到傍晚 $6:00$ 之间陨石首次落下的概率是多少？

假定 $X$ 是为了观察陨石落下所需要的等待时间。由于 $X$ 满足指数型分布，均值为 $\frac{1}{\lambda}=10$ ，由此得：$\lambda=\frac{1}{10}$ 。所求的概率为：
$$
P(\frac{1}{4}\le X \le \frac{3}{4})=P(X\ge \frac{1}{4})-P(X\ge \frac{3}{4})=e^{-\frac{1}{40}}-e^{-\frac{3}{40}}=0.0476
$$
求解这个过程中利用了连续型随机变量 $P(X\ge a)=P(X> a)=e^{-\lambda a}$ 。



### 指数随机变量的无记忆性

一个灯泡的使用寿命 $T$ 是一个指数随机变量，其参数为 $\lambda$ 。Ariadne 将灯打开后离开房间，在外面呆了一段时间以后（时间长度为 $t$），他回到房间后，灯还亮着。这相当于事件 $A=\{T>t\}$ 发生了。记 $X$ 为灯泡的剩余寿命，问 $X$ 的分布函数是什么？

解答：

实际上 $X$ 是在 $A$ 发生条件下的寿命，得到：
$$
\begin{eqnarray}
P(X> x|A) &=& P(T>t+x|T>t) \\
&=& \frac{P(T>t+x, T > t)}{P(T>t)} \\
&=& \frac{P(T> t+x)}{P(T>t)} \\
&=& \frac{e^{-\lambda(t+x)}}{e^{-\lambda t}}\\
&=& e^{-\lambda x}
\end{eqnarray}
$$
灯泡的剩余寿命 $X$ 的分布函数是指数分布，其参数也是 $\lambda$ ，这和灯泡已经亮了多少个小时是无关的。指数分布的这个性质就是**分布的无记忆性**。**一般地，若将完成某个任务所需要的时间的分布定位指数分布。那么只要这个任务没有完成，那么要完成这个任务所需要的剩余时间的分布仍然是指数分布，并且其参数也是不变化的**。

#### 应用

一个粗心的教授错误地将两个学生的答疑时间安排在了同一个时间段。已知两位同学的答疑时间长度是两个相互独立的并且同分布的随机变量，分布是指数分布，期望值为 $30$ 分钟，第一个学生按时到达，5分钟以后，第二个学生也到达。从第一个学生到达起直到第二个学生离开所需要时间的期望值是？

解答：

用 $T_{total}$ 表示教授答疑总共用时的随机变量，用 $T_{s_1}, T_{s_2} $ 表示教授分别对学生 $1$ 和学生 $2$ 答疑时间，那么
$$
E[T_{total}]=P(T_{s_1}< 5) \cdot E[5+E[T_{s_2}]] + P(T_{s_1} \ge 5)(E[T_{s_1}|T_{s_1}\ge5]+E[T_{s_2}])
$$
据题目得：$E[T_{s_1}]=E[T_{s_2}]=30$ ，利用指数型随机变量的无记忆性得到：$E[T_{s_1}|T_{s_1}\ge5] = 5+E[T_{s_1}] = 35$ 。
$$
P(T_{s_1} \ge 5) = e^{-\frac{1}{30}\cdot5} \\
P(T_{s_1} < 5)=1-P(T_{s_1}\ge 5)=1-e^{-\frac{1}{30}\cdot5}\\
$$
因此：
$$
E[T_{total}]=(1-e^{-\frac{1}{30}\cdot5})\cdot (5+30)+(e^{-\frac{1}{30}\cdot5})\cdot (35+30)=35+30\cdot e^{-\frac{5}{30}}=60.394
$$

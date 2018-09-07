---
title:  经典摘录 - 伯努利和泊松随机变量的均值方差
mathjax: true
mathjax2: true
categories: 中文
tags: [probability]
date: 2017-08-28 20:16:00
commets: true
toc: true
---

### Example 2.5. Mean and Variance of the Bernoulli.伯努利随机变量的均值和方差

Consider the experiment of tossing a coin, which comes up a head with probability $p$ and a tail with probability $1 - p$. and the Bernoulli random variable $X$ with $PMF$ :
$$
p_X(k)=\cases{p, & if $k=1$\\1-p, & if $k=0$}
$$
The mean. second moment. and variance of $X$ are given by the following calculations: 
$$
E[X]=1\cdot p + 0 \cdot (1-p) = p ,\\
E[X^2]=1^2\cdot p + 0 \cdot (1-p) = p, \\
var(x)=E[X^2]-(E[X])^2=p-p^2=p(1-p)
$$

### Example 2.7. The Mean of the Poisson. 泊松随机变量的均值方差

The mean of the Poisson $PMF $ :
$$
p_X(k)=e^{-\lambda}\frac{\lambda^k}{k!}, k=0,1,2,\ldots
$$
can be calculated is follows: 
$$
\begin{eqnarray}
E[X] &=& \sum\limits^{\infty}_{k=0}ke^{-\lambda}\frac{\lambda^k}{k!}\\
&=& \sum\limits^{\infty}_{k=1}ke^{-\lambda}\frac{\lambda^k}{k!} \quad (\text{k=0的项等于0})\\
&=& \lambda\sum\limits_{k=1}^{\infty}e^{-\lambda}\frac{\lambda^{k-1}}{k-1!}\\
&=& \lambda\sum\limits_{m=0}^{\infty}e^{-\lambda}\frac{m^{k-1}}{m!} \quad (\text{让m=k-1})\\
&=& \lambda
\end{eqnarray}
$$
The last equality is obtained by noting that is the normalization property for the Poisson $PMF$. 

### Example 2.20. Variance of the Binomial and the Poisson. 二项随机变量和泊松随机变量的方差

We consider $n$ independent coin tosses, with each toss having probability $p$ of coming up a head. For each $i$, we let $X_i$ be the Bernoulli random variable which is equal to $1$ if the $i$th toss comes up a head, and is 0 otherwise. Then, $X = X_l + X_2 + . . . + X_n$ is a binomial random variable. Its mean is $E[X] = np$. as derived in Example 2. 10. By the independence of the coin tosses. the random variables $X_1 , . . . . X_n$ are independent, and 
$$
var(X)=\sum\limits_{i=1}^{n}var(x_i)=np(1-p)
$$
As we discussed in Section 2.2. a Poisson random variable $Y$ with parameter $\lambda$ can be viewed as the "limit" of the binomial as $n\rightarrow \infty, p\rightarrow 0$. while $np = \lambda$. Thus, taking the limit of the mean and the variance of the binomial. we informally obtain the mean and variance of the Poisson: $E[Y] = var(Y) = \lambda $ .  We have indeed verified the formula $E[Y] = \lambda$ in Example 2.7. To verify the formula $var(Y) = \lambda$, we write 
$$
\begin{eqnarray}
E[Y^2] &=& \sum\limits_{k=1}^{\infty}k^2e^{-\lambda}\frac{\lambda^k}{k!} \\
&=& \lambda\sum\limits_{k=1}^{\infty}k\frac{e^{-\lambda}\lambda^{k-1}}{(k-1)!} \\
&=& \lambda\sum\limits_{m=0}^{\infty}(m+1)\frac{e^{-\lambda}\lambda^{m}}{m!}, m = k-1 \\
&=& \lambda[\sum\limits_{m=0}^{\infty}m\frac{e^{-\lambda}\lambda^{m}}{m!}+\sum\limits_{m=0}^{\infty}\frac{e^{-\lambda}\lambda^{m}}{m!}], \\
&=& \lambda(E[Y]+1) , \\
&=& \lambda(\lambda+1) \\
\end{eqnarray}
$$

from which 
$$
var(Y)=E[Y^2]-(E[Y])^2=\lambda(\lambda+1)-\lambda^2=\lambda
$$

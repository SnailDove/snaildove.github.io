---
title: Tsinghua linear-algebra-2 10th-lecture Fourier series
mathjax: true
mathjax2: true
categories: 中文
tags: [linear_algebra]
date: 2017-08-02 20:16:00
toc: true
comments: true
---

笔记源自：清华大学公开课：线性代数2——第10讲：傅里叶级数

## 引言

![preface](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/1.png)

## 傅里叶级数Fourier series定义

### 定义1

设$f(x)$是周期为$2\pi$的有限个分段（[piecewise](https://en.wikipedia.org/wiki/Piecewise)）的连续函数（ continuous function）（即在$[\pi,-\pi]$中只有有限个点不连续，且不连续点的左右极限存在），那么它的傅里叶级数是 $F={a_0\over 2}+\sum\limits_{k=1}^{\infty}(\ a_kcos(kx)+b_ksin(kx)\ ), a_k={1\over \pi}\int_{-\pi}^{\pi}f(x)cos(kx)dx, b_k={1\over \pi}\int_{-\pi}^{\pi}f(x)sin(kx)dx,k=0,1,\ldots$，这个级数又称为**傅里叶级数的实形式**。

$f(x)$举例如下的$(1)$，而 $(2)$ 在周期内的不连续点处无极限。
![example_of_2_kinds_of_piecewise_function](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/2.png)

### 定义2

$f(x)$如上，它的**傅里叶级数的复形式**是$F=\sum\limits_{k=-\infty}^{+\infty}c_ke^{ikx}, c_k={1\over 2\pi}\int_{-\pi}^{\pi}f(x)e^{-ikx}dx$. 推导如下：

在定义1中，使用欧拉公式：$e^{ix}=cosx+isinx\Rightarrow cosx={e^{ix}+e^{-ix}\over 2},\ sinx={e^{ix}-e^{-ix}\over 2i}$ ，定义1中的傅里叶级数变成$F={a_0\over 2}+\sum\limits_{k=1}^{\infty}[{a_k\over 2}(e^{ikx}+e^{-ikx})-{ib_k\over 2}(e^{ikx}-e^{-ikx})]={a_0\over 2}+\sum\limits_{k=1}^{\infty}({a_k-ib_k\over 2}e^{ikx}+{a_k+ib_k\over 2}e^{-ikx}).$ 其中
$a_k-ib_k={1\over 2\pi}\int_{-\pi}^{\pi}f(x)(e^{ikx}+e^{-ikx})dx-{i\over 2\pi}\int_{-\pi}^{\pi}f(x)\frac{e^{ikx}-e^{-ikx}}{i}dx ={1\over \pi}\int_{-\pi}^{\pi}e^{-ikx}dx,\ a_k+ib_k={1\over \pi}\int_{-\pi}^{\pi}e^{ikx}dx​$  , 令$c_k={a_k-ib_k\over 2}={1\over 2\pi}\int_{-\pi}^{\pi}f(x)e^{-ikx}dx,k=1,2,\ldots\quad c_{-k}={a_k+ib_k\over 2}={1\over 2\pi}\int_{-\pi}^{\pi}f(x)e^{ikx}dx,k=1,2,\ldots​$ 这样就得到定义2。注意：正如泰勒级数，这里并没有断言$f(x)​$等于它的傅里叶级数。

### 定理

设$f(x)$是周期为$2\pi$的周期函数，$f(x)$和$f'(x)$均在$[-\pi, \pi]$上是分段连续的，则$f(x)$的傅里叶级数收敛，且在任意连续点$x=a$等于$f(a)$，在不连续点$x=a$等于${1\over 2}[lim_{x\rightarrow a^{+}}f(x)+lim_{x\rightarrow a^{-}}f(x)]$。 

![one_law_of_Fourier_series](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/3.png)

## 内积空间inner product space

设$V$是一个向量空间（线性空间）（$R$或$C$上），$V$上的一个**内积**是这样一个函数 (-,-) : $V\times V\rightarrow R\ or\ C$ 满足：

1.  $\forall u\in V, (u,u)\ge0$，且若$(u,u)=0\rightarrow u=0$ 
2.  $(c_1u+c_2v,w)=c_1(u,w)+c_2(v,w), u,v,w\in V, c_1,c_2\in R\ or\ C$
3.  $\overline{(u,v)}=(v,u)$ 共轭对称

注：没有假设$v$是有限维的。第一条：$u$跟自己的内积必须是一个实数且是一个正数，或者说更确切地是一个非负数，如果$u$跟自己的内积是等于0的，那么就可以确定$u$就是$0$向量。第二条：两个向量的线性组合跟另一个向量的内积相当于两个向量跟另一个向量先作内积再做线性组合。第三条：$u$和$v$的内积与$v$和$u$的内积是一个共轭的关系，如果这个函数是定义在$V\times V\rightarrow R$，那么这个内积函数是个对称的，$u,v$的内积与$v,u$的内积是一样的，如果定义在复数上，那么就差一个共轭。

令$||u||=\sqrt{(u,u)}$，若$||u||=1$，则$u$ 是一个单位向量。任何一个向量$u\ne 0\rightarrow {v\over ||v||}$是一个单位向量，关于[范数]($||·||$)($||·||$)，这里范数是长度。

### 例

1.  $V=R^2,u=\begin{pmatrix}a_1\\a_2\end{pmatrix}, v=\begin{pmatrix}b_1\\b_2\end{pmatrix},(u,v)=u^T v=a_1b_1+a_2b_2$ 是一个内积，$||u||=\sqrt{(a_1^2+a_2^2)}$。若$V=C^2$，$u,v\in C, (u,v)=u^T\bar{v}=a_1\overline{b_1}+a_2\overline{b_2}$ 。
2.  $C[a,b]$是定义在区间$[a,b]$上的全体连续实函数构成的向量空间。定义连续函数的内积为$(f,g)=\int_{a}^{b}f(x)g(x)dx$ 。验证这个式子：$f(x)\in C[a,b], (f,f)\ge 0$ ，即 $(f,f)=\int_{a}^{b}{f(x)}^2dx=\int_{a}^{b}{|f(x)|}^2dx\ge 0$。若$(f,f)=0$，即$\int_{a}^{b}{|f(x)|}^2dx=0$，令$F(t)=\int_{a}^{t}{|f(x)|}^2dx, a\le t \le b$，则$F(t)=0, F(t)$可导，$F'(t)={|f(t)|}^2=0$，即$f(t)=0,t\in [a,b]$。在这里函数的长度的平方定义为函数与自身的内积，即$||f(x)||^2=(f(x),f(x))=\int_{a}^{b}f(x)f(x)dx$。
3.  在例2中，若$C[a,b]$是$[a,b]$上的连续复函数的向量空间，则内积定义为：$(f,g)=\int_{a}^{b}f(x)\overline{g(x)}dx$。

### 标准正交系orthonormal system

![orthonormal system](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/4.png)
总结：若f(x)在区间$[a,b]$存在傅里叶级数，那么f(x)的傅里叶级数是f(x)在标准正交系
$\{\frac{1}{\sqrt{2\pi}}, \frac{1}{\sqrt{\pi}}sinx, \frac{1}{\sqrt{\pi}}cosx, \frac{1}{\sqrt{\pi}}sin2x, \frac{1}{\sqrt{\pi}}cos2x, \ldots\}$下的投影。

## 周期函数的傅里叶级数

对傅里叶级数的实数形式$F={a_0\over 2}+\sum\limits_{k=1}^{\infty}(\ a_kcos(kx)+b_ksin(kx)\ ), a_k={1\over \pi}\int_{-\pi}^{\pi}f(x)cos(kx)dx, b_k={1\over \pi}\int_{-\pi}^{\pi}f(x)sin(kx)dx,k=0,1,\ldots$进行变量代换，令 $x={\pi\over L}t, k=n$ 得：$ dx={\pi\over L}dt,\ t=\cases{L, x=\pi\\ -L, x=-\pi}\Rightarrow f(t)={a_0\over 2}+\sum\limits_{n=1}^{\infty}[a_ncos({n\pi t\over L})+b_nsin({n\pi t\over L})],\ a_n={1\over L}\int_{-L}^{L}f(t)cos({n\pi t\over L})dt, \\b_n={1\over L}\int_{-L}^{L}f(t)sin({n\pi t\over L})dt, n=0,1,\ldots$，对应的复数形式为： 

![Fourier-series_of_periodic_function](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/5.png)

### 投影

![application_of_projection_on_Fourier-series](http://pne0wr4lu.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-10/6.png)
注：$e^{ikx}=cos(kx)+isin(kx)\rightarrow (e^{ikx},e^{ikx})=2L, L$为半周期的绝对值，另外根据复函数的向量空间的内积定义为：$(f,g)=\int_{a}^{b}f(x)\overline{g(x)}dx \rightarrow (f(x),e^{ikx})$在周期 $[-\pi,\pi]$ 下为 $\int_{-\pi}^{\pi}f(x)e^{-iks}dx$ 。

## 关于傅里叶变换的注记

Fourier series傅里叶级数和Fourier transformation傅里叶变换是傅里叶分析的主要部分。设$f(t)$周期$T=2L$，则$f(t)$的傅里叶级数展开为$f(t)=\sum\limits_{k=-\infty}^{\infty}c_ke^{\frac{ik\pi}{L}t},c_k={1\over 2L}\int_{-L}^{L}f(t)e^{\frac{-ik\pi}{L}t}dt$ ($c_k$是$f(t)$在$e^{\frac{ik\pi}{L}t}$上的投影)。现在考虑**定义在$(-\infty, +\infty)$上的非周期函数$f(t)$**，它有傅里叶级数展开形式吗？

给定$L>0$，定义$f_L(t)=\cases{f(t), |t|<L\\0 , \quad\ |t| \ge L}$。假设$L\rightarrow \infty$时，$f_L(t)$（一致）趋近于$f(t)$。函数 $f_L(t)$ 能被周期延拓，即令$F_L(t)=\cases{f(t), -L< t \le L\\ F_L(t+2L), T=2L}$ 则 $F_L(t)$有傅里叶级数。

当 $-L<t<L,f(t)=f_L(t)=F_L(t)=\sum\limits_{k=-\infty}^{\infty}c_k(L)e^{\frac{ik\pi}{L}t},c_k(L)={1\over 2L}\int_{-L}^{L}f_L(t)e^{\frac{-ik\pi}{L}t}dt$

因为 $f_L(t)=0, |t|>L \rightarrow c_k(L)={1\over 2L}\int_{-L}^{L}f_L(t)e^{\frac{-ik\pi}{L}t}dt={1\over 2L}\int_{-\infty}^{\infty}f_L(t)e^{\frac{-ik\pi}{L}t}dt$

由于 $k\rightarrow\infty$ 同时 $L\rightarrow \infty$ ，所以等式右边的指数项未知，因此做变量代换，令 $\tilde{f}_L(w)=\int_{-\infty}^{\infty}f_L(t)e^{-iwt}dt$，令 $w_k={k\pi\over L}$ 则$c_k(L)={1\over 2L}\tilde{f}(\frac{k\pi}{L})={1\over 2L}\tilde{f}({w_k})={1\over 2\pi}\tilde{f}({w_k})(w_{k+1}-w_k)$

那么得到**傅里叶展开的新形式**：$f_L(t)=F_L(t)={1\over 2\pi}\sum\limits_{-\infty}^{+\infty}\tilde{f}_L(w_k)e^{iw_kt}\Delta w_k, \tilde{f}_L(w)=\int_{-\infty}^{+\infty}f_L(t)e^{-iw_kt}dt, \Delta w_k=w_{k+1}-w_{k}={\pi\over L}$。当$L\rightarrow +\infty, \Delta w\rightarrow 0$，等式左边$f_L(t)$（一致）趋近于$f(t)$，右边就趋近于一个积分形式：$ f(t)={1\over 2\pi}\int_{-\infty}^{+\infty}\tilde{f}_L(w)e^{iwt}dw$， 称$\tilde{f}(w)$是$f(t)$的**傅里叶变换**，$f(t)$是$\tilde{f}(w$)的**逆傅里叶变换**。

$f(t)$实际上是关于时间函数的$sin\  cos$之间叠加出来的，那么$\tilde{f}(ω)$是关于这些频率叠加出来的，它是频率的函数。讲复矩阵的时候将会回到这个傅里叶变换，会考虑傅里叶变换的离散形式，那么$f(x)$或者$f(t)$就被一个向量替换，$\tilde{f}(ω)$也被一个向量替换，它们之间互逆的这种傅里叶变换或者逆傅里叶变换的关系，实际上就是通过一个傅里叶矩阵进行互相转换的，以及相应的快速的傅里叶变换。

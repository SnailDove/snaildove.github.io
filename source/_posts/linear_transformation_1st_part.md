---
title: Tsinghua linear-algebra-2 4th-lecture linear-transformation-1st-part
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-04 20:16:00
tags: [linear_algebra]
toc: true
---

笔记源自：清华大学公开课：线性代数2——第四讲：线性变换1

## 前言

历史上英国数学家Arthur Cayley是为了描述线性变换的复合而引入矩阵的乘法，从而使矩阵成为数学的研究对象。线性变换是两个向量空间之间保持线性运算的映射。线性代数就是从其中心问题（求解线性方程组）出发发展起来研究向量空间、线性变换以及研究相关数学问题的数学学科。对有限维向量空间的研究总可以转化成对矩阵的研究，这是线性代数的核心特点。

## 线性变换的定义性质运算

回顾中学阶段学过的函数：$f(x)=2x\quad g(x)=x^{2}\quad l(x)=sin(x)$ 都是一个映射从定义域中的一个数映成值域中的一个数。推广到把向量映射到向量的映射比如f是从 $R^{3}$ 映到 $R^{2}$ 的一个映射：$f:\begin{pmatrix}x\\y\\z\end{pmatrix}\,\rightarrow\,\begin{pmatrix}2x\\3y-z\end{pmatrix}$，我们关心向量空间到向量空间的映射。人们发现平面上的点、空间中的点 、矩阵多项式函数、连续函数等等集合看上去不同但是它们各自的加法和数乘满足同样的性质，于是就引入了向量空间这样的一个抽象的概念来统一地研究向量空间的概念。

### 向量空间的定义

![definition_of_vector_space](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/1.png)

### 线性变换的定义

![definition_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/2.png)

#### 例子

![example1_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/3.png) 
![example2_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/4.png) 
![example3_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/5.png) 
![example4_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/6.png) 
![example5_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/7.png) 
![example6_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/8.png)

**注意，由线性变换的定义 $T:V\,\rightarrow\,W$ 得到 $T(0)=0$**

![example7_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/9.png) 
![example8_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/10.png)

### 线性变换的性质

![properties_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/11.png)

针对第一条证明： 如果 $T(0)\ne0$ 不满足线性变换定义 $T(cx)=cT(x)$，例如： $T(0)=1\,\rightarrow\,T(0)=T(c0)=1\,\ne\,cT(0)=c$

针对第三条证明：若 $x_{1},\,...\,,x_{n}$ 线性相关，那么存在不全为0的数 $c_{1},\,...\,,c_{n}$ 满足 $c_{1}x_{1}\,+\,...\,+\,c_{n}x_{n}=0$  即 $T(c_{1}x_{1}\,+\,c_{2}x_{2}\,+\,...\,+\,c_{n}x_{n})=T(0)=c_{1}f(x_{1})\,+\,...\,+\,c_{n}f(x_{n})=0$，即$T(x_{1}),\,...\,,T(x_{n})$ 线性相关。

### 线性变换的运算

#### 加法

![addition_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/12.png)

#### 数乘

![scalar_multiplication_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/13.png)

#### 乘积

**注：线性变换的乘积被定义为线性变换的复合运算**

![multiplication_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/14.png)

**注意：线性变换不满足乘法交换律、消去律，与矩阵乘法类似**

#### 逆

![inverse_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/15.png)

#### 幂

![exponential_operation_of_linear_transformation1](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/16.png)

#### 多项式

![polynomial_of_linear_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/17.png)

**注：由于线性变换不满足乘法交换律，因此$(\sigma\tau)^{m}=\underbrace{(\sigma\tau)(\sigma\tau)\,...\,(\sigma\tau)}_{m个(\sigma\tau)相乘}\ne\sigma^{m}\tau^{m}$**

## 线性变化的矩阵表示

![线性变换的矩阵表示1](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/18.png) 

由于 $T(v_{1})$,$T(v_{2})$, ... , $T(v_{3})\,\epsilon\,W$ 这个输出空间, 因此可以进行如下： 
![线性变换的矩阵表示2](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/19.png) 
![线性变换的矩阵表示3](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/20.png) 

### 例子

![线性变换的矩阵表示-例子1](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/21.png) 
![线性变换的矩阵表示-例子2](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/22.png) 
![线性变换的矩阵表示-例子3](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/23.png) 
![线性变换的矩阵表示-例子4](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/24.png) 

## 线性变换与矩阵之间的关系

### 一一对应

![the_relation_between_transformation_and_matrix](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/25.png) 
![the_inverse_of_transformation_and_matrix](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/26.png)

### 线性变换的乘积与矩阵的乘积

![the_product_of_transfromation_and_matrices](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/27.png)

**注（极其重要）：这里线性变换的乘积（复合）对应的是矩阵的“左乘”。**

### 线性同构

![linear_isomorphism_between_matrix_and_transformation](http://q3rrj5fj6.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-4/28.png)

**例**：设线性变换$\tau\,:\,R^{3}\rightarrow\,R^{2}$定义为$\tau(x,y,z)=(x+y,y-z)$, 线性变换$\sigma:R^{2}\,\rightarrow\,R^{2}$定义为$\sigma(u,v)=(2u-v,u)$.求线性变换$\sigma\tau:R^{3}\,\rightarrow\,R^{2}$在$R^{3}$与$R^{2}$标准基下的矩阵.

**解**：注意到$\sigma\tau=\sigma(\tau(x,y,z))=\sigma(x+y, y-z)=(2x+y+z, x+y)$ 

因此标准基下线性变化$\sigma(\tau(x\,y\,z)):R^{3}\to\,R^{2}$:

$$e_{1}=(1,0,0)^{T}, e_{2}=(0,1,0)^{T}, e_{3}=(0,0,1)^{T}\,\Rightarrow\, I_{3}=(e_{1}\,e_{2}\,e_{3})$$

$\sigma(\tau(e_{1}))=\sigma(\tau(\,(1,0,0)\,)=\begin{pmatrix}2\\1\\\end{pmatrix}\quad\sigma((\tau(e_{2}))=\begin{pmatrix}1\\1\\\end{pmatrix}\quad\sigma(\tau(e_{3}))=\begin{pmatrix}1\\0\\\end{pmatrix}$

$\sigma(\tau(e_{1}\,e_{2}\,e_{3}))=\sigma(\tau(I_{3}))=\underbrace{\begin{pmatrix}2&1&1\\1&1&0\end{pmatrix}}_{C}$

第一个线性变化$\tau(x,y,z)=(x+y,y-z):R^{3}\,\to\,R^{2}$ :

$$\tau(e_{1})=\tau(1,0,0)=(1+0,0+0)=(1,0)$$

$$\tau(e_{2})=\tau(0,1,0)=(0+1,1+0)=(1,1)$$

$$\tau(e_{3})=\tau(0,0,1)=(0+0,0+1)=(0,1)$$

$$\tau(I_{3})=\tau(e_{1}\,e_{2}\,e_{3})=\begin{pmatrix}1&1&0\\0&1&-1\end{pmatrix}=I_{2}\begin{pmatrix}1&1&0\\0&1&-1\end{pmatrix}$$

$$\underbrace{\begin{pmatrix}1&1&0\\0&1&-1\end{pmatrix}}_{A}\begin{pmatrix}x\\y\\z\end{pmatrix}=\begin{pmatrix}x+y\\y-z\end{pmatrix}$$

第二个线性变化$\sigma(u,v)=(2u-v,u): R^{2}\,\to\,R^{2}$:

$$\delta_{1}=(1,0)^{T}, \delta_{2}=(0,1)^{T}\,\Rightarrow\, I_{2}=(\delta_{1}\,\delta_{2})$$

$$\sigma(\delta_{1})=\begin{pmatrix}2\\1\end{pmatrix},\,\sigma(\delta_{2})=\begin{pmatrix}-1\\0\end{pmatrix}\Rightarrow\sigma(\delta_{1}\,\delta_{2})=I_{2}\begin{pmatrix}2&-1\\1&0\end{pmatrix}$$

$$\underbrace{\begin{pmatrix}2&-1\\1&0\end{pmatrix}}_{B}\begin{pmatrix}u\\v\end{pmatrix}=\begin{pmatrix}2u-v\\u\end{pmatrix}$$

发现$BA=C\,\Rightarrow\,\begin{pmatrix}2&-1\\1&0\end{pmatrix}\begin{pmatrix}1&1&0\\0&1&-1\end{pmatrix}=\begin{pmatrix}2&1&1\\1&1&0\end{pmatrix}$，符合上文所说的线性变换的复合是对应矩阵的左乘。

结论：**有限维向量空间上的线性变换$\leftarrow\rightarrow$矩阵**




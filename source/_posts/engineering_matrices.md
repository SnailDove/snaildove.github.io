---
title: Tsinghua linear-algebra-2 7th-lecture engineering-matrix
mathjax: true
mathjax2: true
categories: 中文
tags: [linear_algebra]
date: 2017-08-07 20:16:00
commets: true
toc: true
---

笔记源自：清华大学公开课：线性代数2——第七讲：工程中的矩阵

## 应用数学的几个原则

1.  将非线性问题变成线性问题(Nonlinear becomes linear)
2.  将连续问题转化为离散的(Continuous becomes discrete)

## 工程中的矩阵

许多物理定理都是线性关系(as approximations of reality)，比如胡克定律(Hook's law)、欧姆定律(Ohm's law)、牛顿第二定律(F=ma)，讨论这些定律的向量形式。线性关系的向量形式以如下**方式、框架**来讨论：

$(1)\ e=Au$
$(2)\  y=Ce$
$(3)\ f=A^Tw$

其中$u$是起始未知量$(primary\ unknown)$，$f$是外部的输入$(input)$：

![framework_of_linear_ralation](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/1.png)

线性问题通常是：输入$f$，求出$u$？

例如：
胡克定律：Displacement is proportional to force  f=ku
欧姆定律：Current is proportional to voltage difference推广到向量形式：$f=Ku$
另外的例子：最小二乘法 $A^TAx=A^Tb$ ，求$x$

![framework_of_least_squares_approximations](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/2.png)

## 线性弹簧模型

![Line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/3.png)

### 情形(1)

![1st_example_of_line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/4.png)

### 情形(2)

![2nd_example_of_line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/5.png)

### 情形(3)

![3rd_example_of_line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/6.png)

### 情形(4)

![4th_example_of_line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/7.png)

### 总结

![summary_of_example_of_line_spring_model](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/8.png)

胡克定律的向量形式把它应用到了弹性力学中，这个$u$表示的是质体的上下位移$e$是弹簧和伸长或缩短量，那么它们之间的关系呢？可以通过这样$A$这个矩阵那么A非常相似于一个差分矩阵，这样弹簧的伸长和缩短量和弹簧的弹力之间可以通过胡克定律来描述，那么这若干根弹簧它们所产生的弹力我们提升到胡克定律这样一个向量形式：C的每一个对角分量表示的是一个弹性系数（$y=Ce$）。最后弹力和外力之间：当达到平衡以后，可以通过一个矩阵去描述它们的关系，这个矩阵跟前面这个矩阵正好互为转置最后把整个过程合起来到这个矩阵$K=A^TCA$，称为刚度矩阵刚度矩阵刻划了系统受外力作用的形变程度。

### 刚度矩阵

![4_kinds_of_stiffness_matrix](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/9.png)

**注：此处老师讲解具有小的跳跃性** ，渣渣注释如下：

-   $K,T$是正定的：$C$ 矩阵表示弹性系数是正定的，$K=A^TCA$ ，当 $A$ 可逆的时候，$K$ 与 $C$ 合同的。

>与正定矩阵合同的对称矩阵也是正定的
>
>​	判断是实对称阵是不是正定的第一条判别法：特征值是否全正，是的话则这个实对称矩阵就是正定的。根据惯性定理，由于与正定矩阵（记为$A$）合同的矩阵（记为$B$）其特征值符号与 $A$ 一致且保持对称性，那么$B$ 的特征值也是全正的，因此 $B$ 也是正定的。

-   $B, C$ 是半正定的

>   因为弹性系数矩阵 $C$ 是正定的对角阵$\Rightarrow x^TKx=x^TA^TCAx=x^TA^T ({\sqrt{C}}^T \sqrt{C}) Ax = x^TA^T {\sqrt{C}}^T \sqrt{C} Ax = ||\sqrt{C} Ax||^2 $，因为$A$是奇异的，$x^TKx\ge 0$， 因此K是半正定的。

#### 性质1

![1st_property_of_4_kinds_stiffness_matrix](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/10.png)

注：$f_i$ 是 $i$ 个质题所受的外力，例如：重力，$f_i=m_ig,\ m_i$ 是 $i$ 个质体的质量。

#### 性质2

![2nd_property_of_4_kinds_stiffness_matrix](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/11.png)

注：此处老师直接说一般性结论，$A, \ B$ 都正定的，那么$AB$ 可能不对称，但是$AB$ 存在正特征值。圆盘定理也是直接引用（！！渣渣工科狗表示闻所未闻！！）。

#### 性质3

![3rd_property_of_4_kinds_stiffness_matrix](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/12.png)

#### 从离散到连续

**$f=A^TCAu$**

![discrete becomes continuous](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/13.png)

总结：

$(1)\ e=u_i-u_{i-1}=\Delta u={du\over dx}=Au\\(2)y=Ce=c(x)e(x)\\(3)\ f=-(y_i-y_{i-1})=-\Delta y=-{dy\over dx}=A^Ty$

![discrete becomes continuous](http://q4vftizgw.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-7/14.png)
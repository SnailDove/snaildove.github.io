---
title: Calculus and Differential in Machine Learning
mathjax: true
mathjax2: true
categories: 中文
tags: [Calculus and Differential]
date: 2018-01-28
comments: true
copyright: true
toc: true
top: true
---

**本文会一直随时间持续更新**

## 梯度

### 一个点的切线的斜率与法线的斜率相乘等于-1

证明：斜率 $k_1=tan\theta$，$\theta$ 是倾斜角，对应的法线的倾斜角为 $\theta+90$，那么
{% raw %}$$k_1 * k_2=tan\theta * tan(\theta+90)=tan\theta * (-cot\theta)=-1$${% endraw %}

### 直线的点法式方程

函数 $y=f(x)$ 在点 $x_0$ 处的导数 $f'(x_0)$ 在几何上表示曲线 $y=f(x)$ 在点 $M(x_0,f(x_0))$ 处的切线的斜率，即
$$f'(x_0)=tan\alpha$$
其中$α$ 是切线的倾角.
根据导数的几何意义并应用直线的点斜式方程，可知曲线 $y=f(x)$ 在点 $M(x_0,y_0)$ 处的切线方程为
$$y - y_0=f'(x_0)(x - x_0)$$
过切线 $M(x_0, y_0)$ 且与切线垂直的直线叫做曲线 $y=f(x)$ 在点 $M$ 处的法线.如果 $f'(x_0)≠0$，法线的斜率为 $-\frac{1}{f'(x_0)}$，从而法线方程为
$$y - y_0 = -\frac{1}{f'(x_0)}(x - x_0)$$
切线斜率与法线斜率相乘等于 $-1​$ 。

### 等值线的法向量

**注意**：这部分内容必须先看到下文的梯度定义以后再看
设方程 $f(x, y) = k$ 确定了隐函数 $y=y(x)$ ，将此函数代入回原方程，得恒等式：
$$f(x,y(x))\equiv 0$$
等式两端对 $x$ 求导：
$$f_x \cdot 1 + f_y \cdot y'(x)=0$$
得：$y'(x)=-\frac{f_x}{f_y}$
等值线 $f(x, y) = k$ 在一点 $(x, y)$ 处的法线斜率为：
$$k=-\frac{1}{y'(x)}=\frac{f_y}{f_x}$$
故等值线 $f(x, y)=k$ 在一点 $(x, y)$ 处的**法线向量**为：
$$\{1, \frac{f_y}{f_x}\}\text{ 或 } \{f_x, f_y\}=\nabla f(x, y)$$
这正好是函数 $f(x, y)$ 在 $(x, y)$ 处的梯度。**所以，函数 $f(x, y)$ 在 $(x, y)$ 处的梯度垂直于函数经过该点的等值线。** 因此，**等值线的单位法向量**可表示为：
$$\frac{\nabla f(x, y)}{|\nabla f(x, y)|}$$

### 方向导数

![《高等数学》（下册），同济版](http://pt8q6wt5q.bkt.clouddn.com/gitpages/Calculus_and_Differential/directional_derivative.png)

### 空间直角坐标系

![《高等数学》（下册），同济版](http://pt8q6wt5q.bkt.clouddn.com/gitpages/Calculus_and_Differential/rectangular_coordinate_system.png)

### 梯度

![《高等数学》（下册），同济版](http://pt8q6wt5q.bkt.clouddn.com/gitpages/Calculus_and_Differential/nabla.png)

## 点到（超）平面的距离

**这一块主要运用在SVM中**

### 平面方程

![《高等数学》（下册），同济版](http://pt8q6wt5q.bkt.clouddn.com/gitpages/Calculus_and_Differential/plane_equation.png)

### 平面外一点到平面的距离
为防止大家忘记向量的点积（数量积），先复习数量积，在求解 平面外一点到平面的距离会用到。

![《高等数学》（下册），同济版](http://pt8q6wt5q.bkt.clouddn.com/gitpages/Calculus_and_Differential/the-distance_from_the-point-outside-a-plane_to_the-plane.png)

## 泰勒公式

1. 参考wiki：[Tailor's theorem](https://en.wikipedia.org/wiki/Taylor%27s_theorem#Higher-order_differentiability)

2. 《高等数学》，同济版上册（一元泰勒公式），同济版下册（二元泰勒公式）

---
title: Tsinghua linear-algebra-2 11th-lecture computer-graphics
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-11 20:16:00
tags: [linear_algebra]
commets: true
toc: true
---

笔记源自：清华大学公开课：线性代数2——第11讲：计算机图像

## 引言

熟悉的三维空间的基本变换是：平移(translation)，伸缩(rescaling)，旋转(rotation)，投影(projection)和反射(reflection)。现在一个问题：平移变换只对于点才有意义，因为平移变换会改变点的坐标，可是普通向量没有位置概念，只有大小和方向。那如何区分点和向量呢？这时候引入**齐次坐标系(homogeneous coordinate system)**。

对于任意一个3维空间点$p$的坐标均是参照（相对于）基点（原点）的坐标，可以表示成$p=x\vec e_1+y\vec e_2+z\vec e_3+ O=x\begin{pmatrix}1\\0\\0\end{pmatrix}+y\begin{pmatrix}0\\1\\0\end{pmatrix}+z\begin{pmatrix}0\\0\\1\end{pmatrix}+\begin{pmatrix}0\\0\\0\end{pmatrix}$，然而$\vec{op}=x\vec e_1+y\vec e_2+z\vec e_3$ 是不参照任何东西的，为了在线性代数中统一表示和区分，把$p=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&0\end{pmatrix}\begin{pmatrix}x\\y\\z\\1\end{pmatrix}\quad \vec{op}=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&0\end{pmatrix}\begin{pmatrix}x\\y\\z\\0\end{pmatrix}$， 这时三维空间中的一个点的齐次坐标是$(x,y,z,1)$或$\begin{pmatrix}x\\y\\z\\1\end{pmatrix}$，一个向量的齐次坐标是$(x,y,z,0)$或$\begin{pmatrix}x\\y\\z\\0\end{pmatrix}$，所以平移变换就不是$R^3\rightarrow R^3$。

定义 一个函数$f: R^n \rightarrow R^N $是一个**刚体运动(rigid motion)**，如果$\forall v,w\in R^n, ||f(v)-f(w)||=||v-w||$，即内部的各点间距离不变。定理 $R^3$上的刚体运动是平移，旋转和反射的合成。此时，$f(v)=Av+v_0$，其中$A$是三阶正交阵。三阶正交阵的分类：设$A$是一个三阶正交阵，则存在实可逆阵$P$，$P^{-1}AP=\begin{pmatrix}cos\theta&-sin\theta&0\\sin\theta&cos\theta&0\\0&0&\pm 1\end{pmatrix}=B$，其中$P=(\alpha_1, \alpha_2, \alpha_3)$，根据相似的性质：$|B|=\pm 1\rightarrow |A|=\pm 1$，$A$本身是一个正交阵，因此$A^TA=I_3$。

若$B_{33=1}, AP=PB\rightarrow A\alpha_3=\alpha_3$是一个旋转矩阵，旋转轴是$\alpha_3$所在直线，旋转角度是沿$\alpha_3$方向逆时针转$\theta$角；若$B_{33}=-1, AP=PB\rightarrow A\alpha_3=-\alpha_3$ 是$A$将$\alpha_3$变为$-\alpha_3$，将$\alpha_1,\alpha_2$所在平面逆时针旋转$\theta$角，此时$A$的作用就是镜面反射和旋转，这里镜面指的是x-y平面。

## 平移translation

![translation_isn't_a_linear_transformation](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/1.png)

![homogeneous_coordinate_and_embedding](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/2.png)

## 伸缩rescaling

![rescaling](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/3.png)

## 旋转rotation

### 3个特殊情形

![special_cases_of_rotation_on_3D](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/4.png)

### 一般情形

![a_general_case_of_rotation_on_3D](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/5.png)

### 旋转的性质

![properties_of_rotation](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/6.png)

## 投影projection

![projection_on_3D](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/7.png)

## 反射

![refection_of_imags](http://q9kvrafcq.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-11/8.png)
